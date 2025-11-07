use crate::{
    event_bus::{EventBus, ModEventStream},
    global::global_event_bus,
};

#[derive(Debug)]
struct EventOne {
    pub message: &'static str,
}

#[derive(Debug)]
struct EventTwo {
    pub message: &'static str,
}

#[test]
fn global_event_bus_publishes_and_receives() {
    let bus = global_event_bus();
    bus.clear_type::<EventOne>();

    let stream = bus.subscribe::<EventOne>();

    bus.publish(EventOne {
        message: "测试A",
    });
    bus.publish(EventOne {
        message: "测试B",
    });

    let collected: Vec<&'static str> = stream
        .try_iter()
        .map(|event| event.into_arc().message)
        .collect();

    assert_eq!(collected, vec!["测试A", "测试B"]);
}

#[test]
fn mod_event_stream_supports_individual_delivery() {
    let bus = EventBus::new();
    let streams = ModEventStream::<(EventOne, EventTwo)>::new(&bus);

    bus.publish(EventOne {
        message: "只有事件A",
    });

    let (event_one_result, event_two_result) = streams.try_recv();

    let event_one = event_one_result
        .expect("EventOne should be delivered independently")
        .into_arc();
    assert_eq!(event_one.message, "只有事件A");
    assert!(matches!(event_two_result, Err(flume::TryRecvError::Empty)));

    bus.publish(EventOne {
        message: "只有事件A",
    });

    bus.publish(EventTwo {
        message: "只有事件B",
    });

    bus.publish(EventTwo {
        message: "只有事件B",
    });

    let (mut event_one_queue, mut event_two_queue) = streams.poll();

    for ev in event_one_queue.drain(..) {
        println!("A -> {:?}", ev.into_arc().message);
    }
    
    for ev in event_two_queue.drain(..) {
        println!("B -> {:?}", ev.into_arc().message);
    }

}
