use arc_swap::ArcSwap;
use dashmap::DashMap;
use flume::{RecvError, RecvTimeoutError, Receiver, Sender, TryRecvError, TrySendError};
use std::{
    any::{Any, TypeId}, cell::Cell, fmt, marker::PhantomData, ops::Deref, sync::{
        Arc, OnceLock, Weak, atomic::{AtomicU64, Ordering}
    }, time::Duration
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
            .map(|entry| entry.value().snapshot().len())
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

static GLOBAL_BUS: OnceLock<EventBus> = OnceLock::new();

pub fn global_event_bus() -> &'static EventBus {
    GLOBAL_BUS.get_or_init(EventBus::new)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Ping(u32);

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
}
