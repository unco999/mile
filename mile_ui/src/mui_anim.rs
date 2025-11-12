use glam::{Vec2, Vec4};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AnimProperty {
    Position,
    Size,
    Color,
    Opacity,
    Custom(u32),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum Easing {
    Linear,
    QuadraticIn,
    QuadraticOut,
    QuadraticInOut,
    CubicIn,
    CubicOut,
    CubicInOut,
    QuartIn,
    QuartOut,
    QuartInOut,
    QuintIn,
    QuintOut,
    QuintInOut,
    SineIn,
    SineOut,
    SineInOut,
    ExpoIn,
    ExpoOut,
    ExpoInOut,
    CircIn,
    CircOut,
    CircInOut,
    BackIn,
    BackOut,
    BackInOut,
    ElasticIn,
    ElasticOut,
    ElasticInOut,
    BounceIn,
    BounceOut,
    BounceInOut,
    Custom(u32),
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LoopConfig {
    pub count: Option<u32>,
    pub ping_pong: bool,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            count: Some(0),
            ping_pong: false,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum AnimTargetValue {
    Scalar(f32),
    Vec2([f32; 2]),
    Vec4([f32; 4]),
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnimationSpec {
    pub property: AnimProperty,
    pub from: Option<AnimTargetValue>,
    pub to: AnimTargetValue,
    #[serde(default)]
    pub from_current: bool,
    #[serde(default)]
    pub is_offset: bool,
    pub duration: f32,
    pub delay: f32,
    pub easing: Easing,
    pub loop_config: LoopConfig,
}

impl AnimationSpec {
    pub fn with_defaults(property: AnimProperty, to: AnimTargetValue) -> Self {
        Self {
            property,
            from: None,
            to,
            from_current: false,
            duration: 0.0,
            delay: 0.0,
            easing: Easing::Linear,
            loop_config: LoopConfig::default(),
            is_offset: false,
        }
    }
}

pub struct AnimBuilder {
    spec: AnimationSpec,
}

impl AnimBuilder {
    pub fn new(property: AnimProperty) -> Self {
        Self {
            spec: AnimationSpec::with_defaults(property, AnimTargetValue::Scalar(0.0)),
        }
    }

    pub fn from(mut self, value: impl Into<AnimTargetValue>) -> Self {
        self.spec.from = Some(value.into());
        self.spec.from_current = false;
        self
    }

    pub fn from_current(mut self) -> Self {
        self.spec.from = None;
        self.spec.from_current = true;
        self
    }

    pub fn to(mut self, value: impl Into<AnimTargetValue>) -> Self {
        self.spec.to = value.into();
        self.spec.is_offset = false;
        self
    }

    pub fn to_offset(mut self, value: impl Into<AnimTargetValue>) -> Self {
        self.spec.to = value.into();
        self.spec.is_offset = true;
        self
    }

    pub fn duration(mut self, seconds: f32) -> Self {
        self.spec.duration = seconds.max(0.0);
        self
    }

    pub fn delay(mut self, seconds: f32) -> Self {
        self.spec.delay = seconds.max(0.0);
        self
    }

    pub fn easing(mut self, easing: Easing) -> Self {
        self.spec.easing = easing;
        self
    }

    pub fn loop_count(mut self, count: u32) -> Self {
        self.spec.loop_config.count = Some(count);
        self
    }

    pub fn infinite(mut self) -> Self {
        self.spec.loop_config.count = None;
        self
    }

    pub fn ping_pong(mut self, enabled: bool) -> Self {
        self.spec.loop_config.ping_pong = enabled;
        self
    }

    pub fn build(self) -> AnimationSpec {
        self.spec
    }
}

impl From<f32> for AnimTargetValue {
    fn from(value: f32) -> Self {
        AnimTargetValue::Scalar(value)
    }
}

impl From<Vec2> for AnimTargetValue {
    fn from(value: Vec2) -> Self {
        AnimTargetValue::Vec2(value.to_array())
    }
}

impl From<Vec4> for AnimTargetValue {
    fn from(value: Vec4) -> Self {
        AnimTargetValue::Vec4(value.to_array())
    }
}
