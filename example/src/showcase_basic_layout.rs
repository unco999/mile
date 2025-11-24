use glam::{vec2, vec4};
use mile_db::DbError;
use mile_ui::{
    mui_anim::Easing,
    mui_prototype::{
        BorderStyle, Mui, PanelPayload, StateStageBuilder, UiEventKind, UiPanelData, UiState,
    },
    prelude::RelLayoutKind,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
struct DataTest {
    count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Default)]
pub struct TestCustomData {
    pub count: u32,
}

// Simple parent container with 3 children laid out horizontally via container spec.
pub fn register_basic_layout() -> Result<(), DbError> {
    let _ = Mui::<DataTest>::new("test_container")?
        //我们的UI设计 遵循状态设计  他可以在交互事件中自由的切换状态  比如
        //下次会修复这个bug
        //让我们给他附加 超多面板吧
        //接着上回  我们的面板可以写入实时frag
        //让我们给他加入小面板
        //按grid排列
        .default_state(UiState(0))
        .state(UiState(0), |state| {
            //我们要把这个panel当作一个容器 然后我们去写他的容器配置
            //好吧  因为新的更改 排列似乎出问题了  我需要回头检查一下
            let state = state
                .z_index(4)
                .container_style()
                //小问题 没有设置容器大小 再来看看
                .slot_size(vec2(108.0, 52.0))
                .size_container(vec2(500.0, 500.0))
                .layout(RelLayoutKind::grid([0.0, 0.0]))
                .finish() //这里要退出容器设置的上下文;
                .position(vec2(100.0, 100.0))
                .color(vec4(0.5, 0.7, 0.6, 1.0))
                .border(BorderStyle {
                    color: [0.1, 0.3, 0.2, 1.0],
                    width: 8.0,
                    radius: 0.0,
                })
                //我们在状态里面加入这个接口 加入frag实时计算buffer
                .size(vec2(500.0, 500.0))
                .events() //进入event上下文构建
                .on_event(UiEventKind::Click, |flow| {
                    let mut data_test = flow.payload(); //这里是取出DataTest的可变引用
                    data_test.count += 1; //给他增加值;
                    flow.set_state(UiState(1)); //如果点击 我们切换到状态1
                })
                .finish();
            state
        })
        .build();

    //  让我们创造子面板

    //加24个小面板
    for idx in 0..24 {
        let uuid = format!("demo_entry_{idx}");
        let panel = Mui::<TestCustomData>::new(Box::leak(uuid.into_boxed_str()))?
            .default_state(UiState(0))
            .state(
                UiState(0),
                move |mut state: StateStageBuilder<TestCustomData>| {
                    //访问关系组件 rel  让他依附 一个panel 当作容器
                    state.rel().container_with::<DataTest>("test_container");

                    //通过子类 确定自己要进入的容器  这里指定DataTest 和test_container 就可以绑定了

                    state
                        .z_index(3 - idx)
                        .position(vec2(0.0, 0.0))
                        .color(vec4(0.1, 0.1, 0.1, 1.0))
                        .size(vec2(108.0, 52.0))
                        .border(BorderStyle {
                            color: [0.15, 0.8, 0.45, 1.0],
                            width: 1.0, 
                            radius: 9.0,
                        })
                        .events()
                        .on_event(UiEventKind::Hover, |flow| {
                            flow.position_anim()
                                .from_current()
                                .offset(vec2(0.0, -14.0))
                                .duration(0.18)
                                .easing(Easing::BackOut)
                                .push(flow);
                        })
                        .on_event(UiEventKind::Out, |flow| {
                            //所有的交互操作 全部在flow里面进行
                            flow.position_anim()
                                .from_current()
                                .to_snapshot()
                                .duration(0.22)
                                .easing(Easing::BackIn)
                                .push(flow);
                        })
                        .finish()
                },
            )
            .build()?;
    }
    Ok(())
}
