// pub mod mui;
pub mod mui_anim;
pub mod mui_group;
pub mod mui_prototype;
mod mui_style;
pub mod runtime;
pub mod structs;
// pub mod ui_network;
pub mod util;

pub mod prelude_event {
    #[derive(Debug)]
    pub struct PanelFragExprEvent {
        pub index: u32,
    }
}

pub mod prelude {
    pub use crate::mui_group::*;
    pub use crate::prelude_event::*;
    pub use crate::structs::*;
}
