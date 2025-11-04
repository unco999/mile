pub mod mui;
pub mod mui_build;
pub mod structs;
pub mod ui_network;
pub mod ui_state;

pub mod prelude_event{
    #[derive(Debug)]
    pub struct EventTest{
        pub index:u32
    }
}

pub mod prelude {
    pub use crate::mui::*;
    pub use crate::mui_build::*;
    pub use crate::structs::*;
    pub use crate::ui_network::*;
    pub use crate::ui_state::*;
    pub use crate::prelude_event::*;
}
