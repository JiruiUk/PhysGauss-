# 固定视角时间序列检查工作记录

用户要求检查：多时刻不同相机的训练画面较好，固定同一个相机后，整个时间序列能否保持完整、不破碎。

已完成只读诊断：C1最终50k、T1关联训练52k、O1光流训练50k，各固定训练相机2/86/188并扫描全部200个统一全局时刻，另导出原相机轨迹对照。2400次1600×900原生渲染，16个200帧视频全部解码核验；没有训练或更新模型。C1/T1的相机86/时刻86与此前原生PNG逐像素一致，检查点和输入/源码哈希保持不变。

主要结果：固定相机86在时刻86较清楚，早/晚时刻明显拉丝、碎片和鸭子轮廓丢失；固定相机2晚期失稳，固定相机188早/中期失稳。T1/O1均未消除这一现象。alpha也有不规则支撑和缺口，不能仅归于颜色。原轨迹RGB较好不等于固定视角全时段几何/外观完整。新相机/时刻组合没有GT，未声称真实3D、身份或物理准确率，也未单独确定失稳原因。

- 报告：`experiments/fixed_camera_temporal/assessment.md`
- 验证：`experiments/fixed_camera_temporal/final_verification.json`
- 浏览：`output/fixed_camera_temporal/index.html`
- 关键图：`output/fixed_camera_temporal/key_comparison.png`
- 三模型固定86视频：`output/fixed_camera_temporal/fixed_0086_comparison_RGB_alpha.mp4`
- 导出脚本：`output/fixed_camera_temporal/render_sequences.py`
- 视频和产物核验：`output/fixed_camera_temporal/finalize.py`

人工范围为三模型×三固定视角×八选定时刻的RGB/alpha接触表、C1原轨迹接触表、四条比较视频的解码接触表，以及C1固定86/时刻200原生RGB/alpha；另检查关键对照图。没有逐张人工审阅所有原生图或完整播放所有视频。

后续建议是先定位几何/形变表示失稳，再继续评价遮挡身份。没有据此启动新训练、添加本构/外力、修改相机约定或恢复历史协调器。保留历史模型、缓存和原有三个无关tracked改动；大体积视频和逐帧PNG保留本地，不纳入Git提交。
