use arc_swap::ArcSwap;
use dashmap::DashMap;
use flume::{Receiver, RecvError, RecvTimeoutError, Sender, TryRecvError, TrySendError};
use serde_json::Value as JsonValue;
use std::{
    any::{Any, TypeId},
    cell::Cell,
    fmt,
    marker::PhantomData,
    ops::Deref,
    sync::{
        Arc, Weak,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

pub trait Event: Any + Send + Sync + 'static {}
impl<T: Any + Send + Sync + 'static> Event for T {}

type DynEvent = Arc<dyn Any + Send + Sync + 'static>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OverflowStrategy {
    Block,
    DropNewest,
}

impl Default for OverflowStrategy {
    fn default() -> Self {
        OverflowStrategy::Block
    }
}

#[derive(Clone, Debug, Default)]
pub struct SubscriptionOptions {
    pub label: Option<String>,
    pub buffer: Option<usize>,
    pub overflow: OverflowStrategy,
}

#[derive(Clone)]
struct Subscriber {
    id: u64,
    label: Option<String>,
    sender: Sender<DynEvent>,
    overflow: OverflowStrategy,
}

#[derive(Default)]
struct SubscriberList {
    entries: ArcSwap<Vec<Subscriber>>,
}

impl SubscriberList {
    fn new() -> Self {
        Self {
            entries: ArcSwap::from_pointee(Vec::new()),
        }
    }

    fn snapshot(&self) -> Arc<Vec<Subscriber>> {
        self.entries.load_full()
    }

    fn push(&self, subscriber: Subscriber) {
        self.entries.rcu(|current| {
            let mut next = Vec::with_capacity(current.len() + 1);
            next.extend(current.iter().cloned());
            next.push(subscriber.clone());
            Arc::new(next)
        });
    }

    fn prune(&self, subscriber_id: u64) -> bool {
        let removed = Cell::new(false);
        self.entries.rcu(|current| {
            if current.iter().any(|sub| sub.id == subscriber_id) {
                removed.set(true);
                let filtered: Vec<_> = current
                    .iter()
                    .cloned()
                    .filter(|sub| sub.id != subscriber_id)
                    .collect();
                Arc::new(filtered)
            } else {
                Arc::clone(current)
            }
        });
        removed.get() && self.entries.load().is_empty()
    }
}

#[derive(Default)]
struct EventBusInner {
    subscribers: DashMap<TypeId, Arc<SubscriberList>>,
    next_id: AtomicU64,
}

#[derive(Clone, Default)]
pub struct EventBus {
    inner: Arc<EventBusInner>,
}

impl EventBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish<E: Event>(&self, event: E) -> usize {
        self.publish_arc(Arc::new(event))
    }

    pub fn publish_arc<E: Event>(&self, event: Arc<E>) -> usize {
        let type_id = TypeId::of::<E>();
        let entry = self.inner.subscribers.get(&type_id);
        let Some(entry) = entry else {
            return 0;
        };
        let list = entry.value().clone();
        let snapshot = list.snapshot();
        drop(entry);

        if snapshot.is_empty() {
            return 0;
        }

        let dyn_event: DynEvent = event;
        let mut delivered = 0usize;
        let mut stale = Vec::new();

        for subscriber in snapshot.iter() {
            let res = match subscriber.overflow {
                OverflowStrategy::Block => subscriber
                    .sender
                    .send(dyn_event.clone())
                    .map_err(|err| TrySendError::Disconnected(err.0)),
                OverflowStrategy::DropNewest => subscriber.sender.try_send(dyn_event.clone()),
            };

            match res {
                Ok(()) => delivered += 1,
                Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => stale.push(subscriber.id),
            }
        }

        if !stale.is_empty() {
            for id in stale {
                self.inner.remove_subscriber(type_id, id);
            }
        }

        delivered
    }

    pub fn has_subscribers<E: Event>(&self) -> bool {
        self.subscriber_count::<E>() > 0
    }

    pub fn subscriber_count<E: Event>(&self) -> usize {
        self.inner
            .subscribers
            .get(&TypeId::of::<E>())
            .map(
                |entry: dashmap::mapref::one::Ref<'_, TypeId, Arc<SubscriberList>>| {
                    entry.value().snapshot().len()
                },
            )
            .unwrap_or(0)
    }

    pub fn subscribe<E: Event>(&self) -> EventStream<E> {
        self.subscribe_with_options::<E>(SubscriptionOptions::default())
    }

    pub fn subscribe_with_label<E: Event, S: Into<String>>(&self, label: S) -> EventStream<E> {
        let mut options = SubscriptionOptions::default();
        options.label = Some(label.into());
        self.subscribe_with_options::<E>(options)
    }

    pub fn subscribe_bounded<E: Event>(
        &self,
        capacity: usize,
        overflow: OverflowStrategy,
    ) -> EventStream<E> {
        let mut options = SubscriptionOptions::default();
        options.buffer = Some(capacity);
        options.overflow = overflow;
        self.subscribe_with_options::<E>(options)
    }

    pub fn subscribe_with_options<E: Event>(&self, options: SubscriptionOptions) -> EventStream<E> {
        let type_id = TypeId::of::<E>();
        let buffer = options.buffer.filter(|&size| size != 0);
        let (sender, receiver) = buffer
            .map(|size| flume::bounded::<DynEvent>(size))
            .unwrap_or_else(flume::unbounded);

        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let subscriber = Subscriber {
            id,
            label: options.label.clone(),
            sender,
            overflow: options.overflow,
        };

        self.inner.register_subscriber(type_id, subscriber);

        EventStream {
            receiver,
            type_id,
            subscription_id: id,
            bus: Arc::downgrade(&self.inner),
            _marker: PhantomData,
        }
    }

    pub fn clear_type<E: Event>(&self) {
        self.inner.subscribers.remove(&TypeId::of::<E>());
    }

    pub fn clear_all(&self) {
        self.inner.subscribers.clear();
    }
}

impl EventBusInner {
    fn register_subscriber(&self, type_id: TypeId, subscriber: Subscriber) {
        let list = self
            .subscribers
            .entry(type_id)
            .or_insert_with(|| Arc::new(SubscriberList::new()))
            .clone();
        list.push(subscriber);
    }

    fn remove_subscriber(&self, type_id: TypeId, subscription_id: u64) {
        if let Some(entry) = self.subscribers.get(&type_id) {
            let list = entry.value().clone();
            drop(entry);
            if list.prune(subscription_id) {
                self.subscribers.remove(&type_id);
            }
        }
    }
}

pub struct EventDelivery<E: Event> {
    inner: Arc<E>,
}

impl<E: Event> EventDelivery<E> {
    pub fn into_arc(self) -> Arc<E> {
        self.inner
    }

    pub fn into_owned(self) -> Result<E, Arc<E>>
    where
        E: Sized,
    {
        Arc::try_unwrap(self.inner)
    }
}

impl<E: Event> Deref for EventDelivery<E> {
    type Target = E;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<E> fmt::Debug for EventDelivery<E>
where
    E: Event + fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.inner, f)
    }
}
pub struct ModEventStream<T>
where
    T: EventTuple,
{
    streams: T::Streams,
}

impl<T> ModEventStream<T>
where
    T: EventTuple,
{
    pub fn new(event_bus: &EventBus) -> Self {
        Self {
            streams: T::subscribe(event_bus),
        }
    }

    pub fn from_streams(streams: T::Streams) -> Self {
        Self { streams }
    }

    pub fn into_streams(self) -> T::Streams {
        self.streams
    }

    pub fn streams(&self) -> &T::Streams {
        &self.streams
    }

    pub fn streams_mut(&mut self) -> &mut T::Streams {
        &mut self.streams
    }

    pub fn try_recv(&self) -> T::TryResults {
        T::try_recv(&self.streams)
    }

    pub fn poll(&self) -> T::PollResults {
        T::poll(&self.streams)
    }
}

impl<T> fmt::Debug for ModEventStream<T>
where
    T: EventTuple,
    T::Streams: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModEventStream")
            .field("streams", &self.streams)
            .finish()
    }
}

pub trait EventTuple: Sized {
    type Streams;
    type TryResults;
    type PollResults;

    fn subscribe(event_bus: &EventBus) -> Self::Streams;
    fn try_recv(streams: &Self::Streams) -> Self::TryResults;
    fn poll(streams: &Self::Streams) -> Self::PollResults;
}

macro_rules! impl_event_tuple {
    ($([$event:ident, $binding:ident]),+ $(,)?) => {
        impl<$($event: Event),+> EventTuple for ($($event,)+) {
            type Streams = ($(EventStream<$event>,)+);
            type TryResults = ($(Result<EventDelivery<$event>, TryRecvError>,)+);
            type PollResults = ($(Vec<EventDelivery<$event>>,)+);

            fn subscribe(event_bus: &EventBus) -> Self::Streams {
                ($(event_bus.subscribe::<$event>(),)+)
            }

            fn try_recv(streams: &Self::Streams) -> Self::TryResults {
                let &( $( ref $binding, )+ ) = streams;
                ($( $binding.try_recv(), )+)
            }

            fn poll(streams: &Self::Streams) -> Self::PollResults {
                let &( $( ref $binding, )+ ) = streams;
                ($( $binding.drain(), )+)
            }
        }
    };
}

impl_event_tuple!([E0, stream0]);
impl_event_tuple!([E0, stream0], [E1, stream1]);
impl_event_tuple!([E0, stream0], [E1, stream1], [E2, stream2]);
impl_event_tuple!([E0, stream0], [E1, stream1], [E2, stream2], [E3, stream3]);
impl_event_tuple!(
    [E0, stream0],
    [E1, stream1],
    [E2, stream2],
    [E3, stream3],
    [E4, stream4]
);
impl_event_tuple!(
    [E0, stream0],
    [E1, stream1],
    [E2, stream2],
    [E3, stream3],
    [E4, stream4],
    [E5, stream5]
);

pub struct EventStream<E: Event> {
    receiver: Receiver<DynEvent>,
    type_id: TypeId,
    subscription_id: u64,
    bus: Weak<EventBusInner>,
    _marker: PhantomData<E>,
}

impl<E: Event> EventStream<E> {
    fn downcast(event: DynEvent) -> Arc<E> {
        Arc::downcast::<E>(event).unwrap_or_else(|_| {
            panic!("failed to downcast event to {}", std::any::type_name::<E>())
        })
    }

    #[inline]
    fn wrap(event: DynEvent) -> EventDelivery<E> {
        EventDelivery {
            inner: Self::downcast(event),
        }
    }

    #[inline]
    pub fn recv(&self) -> Result<EventDelivery<E>, RecvError> {
        self.receiver.recv().map(Self::wrap)
    }

    #[inline]
    pub fn recv_timeout(&self, timeout: Duration) -> Result<EventDelivery<E>, RecvTimeoutError> {
        self.receiver.recv_timeout(timeout).map(Self::wrap)
    }

    #[inline]
    pub fn try_recv(&self) -> Result<EventDelivery<E>, TryRecvError> {
        self.receiver.try_recv().map(Self::wrap)
    }

    #[inline]
    pub fn drain(&self) -> Vec<EventDelivery<E>> {
        self.receiver.try_iter().map(Self::wrap).collect()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = EventDelivery<E>> + '_ {
        self.receiver.iter().map(Self::wrap)
    }

    #[inline]
    pub fn try_iter(&self) -> impl Iterator<Item = EventDelivery<E>> + '_ {
        self.receiver.try_iter().map(Self::wrap)
    }

    #[inline]
    pub fn poll_latest(&self) -> Option<EventDelivery<E>> {
        self.receiver.try_iter().map(Self::wrap).last()
    }

    #[inline]
    pub fn recv_arc(&self) -> Result<Arc<E>, RecvError> {
        self.recv().map(EventDelivery::into_arc)
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.receiver.len()
    }
}

impl<E: Event> Drop for EventStream<E> {
    fn drop(&mut self) {
        if let Some(bus) = self.bus.upgrade() {
            bus.remove_subscriber(self.type_id, self.subscription_id);
        }
    }
}

impl<E: Event> fmt::Debug for EventStream<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EventStream")
            .field("type_id", &self.type_id)
            .field("subscription_id", &self.subscription_id)
            .finish()
    }
}

type JsonEvent = Arc<JsonValue>;

#[derive(Clone)]
struct KeyedSubscriber {
    id: u64,
    label: Option<String>,
    sender: Sender<JsonEvent>,
    overflow: OverflowStrategy,
}

#[derive(Default)]
struct KeyedSubscriberList {
    entries: ArcSwap<Vec<KeyedSubscriber>>,
}

impl KeyedSubscriberList {
    fn new() -> Self {
        Self {
            entries: ArcSwap::from_pointee(Vec::new()),
        }
    }

    fn snapshot(&self) -> Arc<Vec<KeyedSubscriber>> {
        self.entries.load_full()
    }

    fn push(&self, subscriber: KeyedSubscriber) {
        self.entries.rcu(|current| {
            let mut next = Vec::with_capacity(current.len() + 1);
            next.extend(current.iter().cloned());
            next.push(subscriber.clone());
            Arc::new(next)
        });
    }

    fn prune(&self, subscriber_id: u64) -> bool {
        let removed = Cell::new(false);
        self.entries.rcu(|current| {
            if current.iter().any(|sub| sub.id == subscriber_id) {
                removed.set(true);
                let filtered: Vec<_> = current
                    .iter()
                    .cloned()
                    .filter(|sub| sub.id != subscriber_id)
                    .collect();
                Arc::new(filtered)
            } else {
                Arc::clone(current)
            }
        });
        removed.get() && self.entries.load().is_empty()
    }
}

#[derive(Default)]
struct KeyedEventBusInner {
    subscribers: DashMap<String, Arc<KeyedSubscriberList>>,
    next_id: AtomicU64,
}

#[derive(Clone, Default)]
pub struct KeyedEventBus {
    inner: Arc<KeyedEventBusInner>,
}

impl KeyedEventBus {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn publish(&self, key: impl AsRef<str>, payload: JsonValue) -> usize {
        self.publish_arc(key, Arc::new(payload))
    }

    pub fn publish_arc(&self, key: impl AsRef<str>, payload: Arc<JsonValue>) -> usize {
        let entry = self.inner.subscribers.get(key.as_ref());
        let Some(entry) = entry else {
            return 0;
        };
        let list = entry.value().clone();
        let snapshot = list.snapshot();
        drop(entry);

        if snapshot.is_empty() {
            return 0;
        }

        let mut delivered = 0usize;
        let mut stale = Vec::new();

        for subscriber in snapshot.iter() {
            let res = match subscriber.overflow {
                OverflowStrategy::Block => subscriber
                    .sender
                    .send(payload.clone())
                    .map_err(|err| TrySendError::Disconnected(err.0)),
                OverflowStrategy::DropNewest => subscriber.sender.try_send(payload.clone()),
            };

            match res {
                Ok(()) => delivered += 1,
                Err(TrySendError::Full(_)) => {}
                Err(TrySendError::Disconnected(_)) => stale.push(subscriber.id),
            }
        }

        if !stale.is_empty() {
            for id in stale {
                self.inner.remove_subscriber(key.as_ref(), id);
            }
        }

        delivered
    }

    pub fn has_subscribers(&self, key: impl AsRef<str>) -> bool {
        self.subscriber_count(key) > 0
    }

    pub fn subscriber_count(&self, key: impl AsRef<str>) -> usize {
        self.inner
            .subscribers
            .get(key.as_ref())
            .map(
                |entry: dashmap::mapref::one::Ref<'_, String, Arc<KeyedSubscriberList>>| {
                    entry.value().snapshot().len()
                },
            )
            .unwrap_or(0)
    }

    pub fn subscribe(&self, key: impl Into<String>) -> KeyedEventStream {
        self.subscribe_with_options(key, SubscriptionOptions::default())
    }

    pub fn subscribe_bounded(
        &self,
        key: impl Into<String>,
        capacity: usize,
        overflow: OverflowStrategy,
    ) -> KeyedEventStream {
        let mut options = SubscriptionOptions::default();
        options.buffer = Some(capacity);
        options.overflow = overflow;
        self.subscribe_with_options(key, options)
    }

    pub fn subscribe_with_label(
        &self,
        key: impl Into<String>,
        label: impl Into<String>,
    ) -> KeyedEventStream {
        let mut options = SubscriptionOptions::default();
        options.label = Some(label.into());
        self.subscribe_with_options(key, options)
    }

    pub fn subscribe_with_options(
        &self,
        key: impl Into<String>,
        options: SubscriptionOptions,
    ) -> KeyedEventStream {
        let key_owned = key.into();
        let buffer = options.buffer.filter(|&size| size != 0);
        let (sender, receiver) = buffer
            .map(|size| flume::bounded::<JsonEvent>(size))
            .unwrap_or_else(flume::unbounded);

        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let subscriber = KeyedSubscriber {
            id,
            label: options.label.clone(),
            sender,
            overflow: options.overflow,
        };

        self.inner
            .register_subscriber(key_owned.clone(), subscriber);

        KeyedEventStream {
            receiver,
            key: key_owned,
            subscription_id: id,
            bus: Arc::downgrade(&self.inner),
        }
    }

    pub fn clear_key(&self, key: impl AsRef<str>) {
        self.inner.subscribers.remove(key.as_ref());
    }

    pub fn clear_all(&self) {
        self.inner.subscribers.clear();
    }
}

impl KeyedEventBusInner {
    fn register_subscriber(&self, key: String, subscriber: KeyedSubscriber) {
        let list = self
            .subscribers
            .entry(key)
            .or_insert_with(|| Arc::new(KeyedSubscriberList::new()))
            .clone();
        list.push(subscriber);
    }

    fn remove_subscriber(&self, key: impl AsRef<str>, subscription_id: u64) {
        if let Some(entry) = self.subscribers.get(key.as_ref()) {
            let list = entry.value().clone();
            drop(entry);
            if list.prune(subscription_id) {
                self.subscribers.remove(key.as_ref());
            }
        }
    }
}

pub struct KeyedEventDelivery {
    inner: Arc<JsonValue>,
}

impl KeyedEventDelivery {
    pub fn into_arc(self) -> Arc<JsonValue> {
        self.inner
    }

    pub fn into_owned(self) -> Result<JsonValue, Arc<JsonValue>> {
        Arc::try_unwrap(self.inner)
    }

    pub fn as_ref(&self) -> &JsonValue {
        &self.inner
    }
}

impl fmt::Debug for KeyedEventDelivery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.inner, f)
    }
}

pub struct KeyedEventStream {
    receiver: Receiver<JsonEvent>,
    key: String,
    subscription_id: u64,
    bus: Weak<KeyedEventBusInner>,
}

impl KeyedEventStream {
    #[inline]
    pub fn recv(&self) -> Result<KeyedEventDelivery, RecvError> {
        self.receiver
            .recv()
            .map(|event| KeyedEventDelivery { inner: event })
    }

    #[inline]
    pub fn recv_timeout(&self, timeout: Duration) -> Result<KeyedEventDelivery, RecvTimeoutError> {
        self.receiver
            .recv_timeout(timeout)
            .map(|event| KeyedEventDelivery { inner: event })
    }

    #[inline]
    pub fn try_recv(&self) -> Result<KeyedEventDelivery, TryRecvError> {
        self.receiver
            .try_recv()
            .map(|event| KeyedEventDelivery { inner: event })
    }

    #[inline]
    pub fn drain(&self) -> Vec<KeyedEventDelivery> {
        self.receiver
            .try_iter()
            .map(|event| KeyedEventDelivery { inner: event })
            .collect()
    }

    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = KeyedEventDelivery> + '_ {
        self.receiver
            .iter()
            .map(|event| KeyedEventDelivery { inner: event })
    }

    #[inline]
    pub fn try_iter(&self) -> impl Iterator<Item = KeyedEventDelivery> + '_ {
        self.receiver
            .try_iter()
            .map(|event| KeyedEventDelivery { inner: event })
    }

    #[inline]
    pub fn poll_latest(&self) -> Option<KeyedEventDelivery> {
        self.receiver
            .try_iter()
            .map(|event| KeyedEventDelivery { inner: event })
            .last()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.receiver.is_empty()
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.receiver.len()
    }
}

impl Drop for KeyedEventStream {
    fn drop(&mut self) {
        if let Some(bus) = self.bus.upgrade() {
            bus.remove_subscriber(&self.key, self.subscription_id);
        }
    }
}

impl fmt::Debug for KeyedEventStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KeyedEventStream")
            .field("key", &self.key)
            .field("subscription_id", &self.subscription_id)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Ping(u32);
    #[derive(Debug)]
    struct Pong(u32);

    #[test]
    fn publish_to_multiple_subscribers() {
        let bus = EventBus::new();
        let sub_a = bus.subscribe::<Ping>();
        let sub_b = bus.subscribe::<Ping>();

        assert_eq!(bus.subscriber_count::<Ping>(), 2);
        assert_eq!(bus.publish(Ping(7)), 2);
        assert_eq!(sub_a.recv().unwrap().into_arc().0, 7);
        assert_eq!(sub_b.recv().unwrap().into_arc().0, 7);
    }

    #[test]
    fn dropping_stream_unsubscribes() {
        let bus = EventBus::new();
        let stream = bus.subscribe::<Ping>();
        assert!(bus.has_subscribers::<Ping>());
        drop(stream);
        assert!(!bus.has_subscribers::<Ping>());
    }

    #[test]
    fn bounded_queue_drops_on_overflow() {
        let bus = EventBus::new();
        let stream = bus.subscribe_bounded::<Ping>(1, OverflowStrategy::DropNewest);

        assert_eq!(bus.publish(Ping(1)), 1);
        assert_eq!(bus.publish(Ping(2)), 0);

        let events = stream.drain();
        assert_eq!(events.len(), 1);
        assert_eq!(events.into_iter().next().unwrap().into_arc().0, 1);
    }

    #[test]
    fn into_owned_transfers_data() {
        let bus = EventBus::new();
        let stream = bus.subscribe::<Ping>();

        bus.publish(Ping(99));

        let event = stream.recv().unwrap();
        let owned = event.into_owned().expect("exclusive ownership");
        assert_eq!(owned.0, 99);
    }

    #[test]
    fn mod_event_stream_handles_multiple_types() {
        let bus = EventBus::new();
        let streams = ModEventStream::<(Ping, Pong)>::new(&bus);

        bus.publish(Ping(1));
        bus.publish(Pong(2));

        let (ping, pong) = streams.try_recv();

        assert_eq!(ping.unwrap().into_arc().0, 1);
        assert_eq!(pong.unwrap().into_arc().0, 2);

        bus.publish(Ping(10));
        bus.publish(Ping(11));
        bus.publish(Pong(12));

        let (mut ping_events, mut pong_events) = streams.poll();

        assert_eq!(ping_events.len(), 2);
        assert_eq!(ping_events.remove(0).into_arc().0, 10);
        assert_eq!(ping_events.remove(0).into_arc().0, 11);

        assert_eq!(pong_events.len(), 1);
        assert_eq!(pong_events.remove(0).into_arc().0, 12);
    }

    #[test]
    fn keyed_event_bus_delivers_by_key() {
        let bus = KeyedEventBus::new();
        let stream_a = bus.subscribe("alpha");
        let stream_b = bus.subscribe("beta");

        assert_eq!(bus.subscriber_count("alpha"), 1);
        assert_eq!(bus.subscriber_count("beta"), 1);

        assert_eq!(bus.publish("alpha", JsonValue::from(1)), 1);
        assert_eq!(bus.publish("beta", JsonValue::from(2)), 1);
        assert_eq!(bus.publish("gamma", JsonValue::from(3)), 0);

        assert_eq!(
            stream_a.recv().unwrap().into_owned().unwrap(),
            JsonValue::from(1)
        );
        assert_eq!(
            stream_b.recv().unwrap().into_owned().unwrap(),
            JsonValue::from(2)
        );
    }

    #[test]
    fn dropping_keyed_stream_unsubscribes() {
        let bus = KeyedEventBus::new();
        let stream = bus.subscribe("alpha");
        assert!(bus.has_subscribers("alpha"));
        drop(stream);
        assert!(!bus.has_subscribers("alpha"));
    }
}
