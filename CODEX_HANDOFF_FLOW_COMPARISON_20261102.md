# Codex 工作交接：运动光流与轨迹导出速度对照

> **最新解释与工作记录（2026-11-02，文档更新）**：用户追问尺度与归因后，已明确：尺度修正仅指诊断400×225→实际训练1600×900，未改时间/3D坐标/速度或参数；**尚未排除C1持续表面表示问题，也不是二维完全无法关联三维点。** 局部关联可建立，但固定行/canonical坐标不证明真实材料ID，后期重新播种不等于早期同一表面。当前卡在长期/跨遮挡的同一物体同一表面证据，模型表示、可见性变化、关联不足与观测歧义尚未分离。独立工作记录 `WORK_RECORD_20261102_SURFACE_CORRESPONDENCE.md` 汇总结果、限制、证据入口及建议；建议尚未执行。本次仅记录，无新训练/模型修改/自动跟进，不恢复旧P1/O0。

> **最新：四清晰短片段冻结C1诊断完成（2026-11-02，零训练/模型更新）**。沿用数据筛选固定2–14/188–200/15–27与嵌套2–8，24唯一训练RGB、21唯一相邻对。三个不同起点都能从前三观测建立四组可见鸭子局部固定canonical库；后续同时检查中心、完整RGB颜色伴随贡献及源/目标原始FB流，未知不强制分配。**关键诊断修正：C1实际训练宽度1600，直接400渲染的PSNR/足迹门槛不能当历史基准或3D错误率。** 全24观测1600训练视图PSNR42.78–49.32dB；1600选点/渲染的后续接受鸭子alpha覆盖率按上述片段顺序为66.85%/37.43%/64.10%/67.61%，不是身份准确率。晚期同一400点库的F0在400略低95%纯度/90%中心门槛，1600通过；F1中心仍边界未过。取消额外整库拒绝但不变逐点门槛的事后分析，晚期34.59%变46.56%，说明严格拒绝会放大边界差异，不作为新算法收益。训练尺度也未恢复所有晚期支持。

> 独立LK与raw流大体接近；静止蓝色结构条件下训练相机极线残差为亚像素，元数据常规相机假设通常更小，但短基线/近似平面/流估计/静止假设不足以证明标定或深度真值。保持C1训练相机原约定，未修相机。蓝色行世界漂移仅条件性表示线索，不归因外力。6测试及独立核验通过：原源码/输入/完整状态与历史C1保护SHA、全贡献积分、固定库/逐点未知/三尺度策略12覆盖率、相机残差、48拼接PNG像素与24独立PNG PSNR量化界。人工看总览/门槛/蓝色运动/LK、四端点接触表与200完整尺寸，非全24逐张审阅。报告 `experiments/short_clip_diagnostic/assessment.md` 与 `verification.json`；浏览 `output/short_clip_diagnostic/index.html`，图 `comparison.png`、`native_review.png`、`late_gates.png`。保留 `runs/pidg_short_clip_diagnostic/cache` 与旧S1全部历史测量。**局部可对应，不代表早期材料表面跨遮挡身份已证实；长期对应仍未通过，不恢复P1、不新开训练或自动跟进。** 先前S1的17.76%是特定400尺度保守规则下接受贡献，不能解释为全模型只有17.76%几何正确；本轮未改旧结果。

> **最新：Rubber Duck清晰短片段数据筛查完成（2026-11-02，数据只读，无模型运行/训练）**。用户要求检查PIDG数据有无低歧义短片段，用当前120训练RGB/原始双向流/相机元信息扫描113个八观测窗口，62通过四轮廓分离/边界/面积/FB纯度与支持检查（窗口重叠，非62独立样本）。不使用C1预测选片，不访问新增验证/测试RGB。首选原2–14，实际训练观测[2,3,6,7,8,10,12,14]，最大相机相对转角2.2835°、最小轮廓间隔7.07px@400；更短2–8五观测控制转角0.6652°；晚期188–200八观测转角2.5595°；15–27八观测6.1223°。相机不是严格固定。首选原图四轮廓分离，上方两只背景清楚；不能证明零3D接触/完整表面无遮挡或真实材料ID，晚期靠蓝色结构只作补充。原图2560×1440，已看三接触表、首选细节及原2/8/14/195四张原生图。报告 `experiments/short_clip_audit/assessment.md`，`{plan,results,input_lock,manual_review,verification}.json`；浏览 `output/short_clip_audit/index.html`，首选clip1_0002_0014.png、细节primary_duck_crops.png。独立验证选片FB/姿态、接触表像素、原生导出像素与输入SHA通过。其他四PIDG场景只确认安装，未同等详查。没有新的C1几何因果诊断、参数修改或训练；这些候选可用于下一步独立相机/局部对应排查。

> **最新：S1 可见片段—固定三维表面库建立并完成全时段核验（2026-11-02，零新训练）**。按用户授权执行图像片段与持续模型表面的连接，独立 `variants/surface_correspondence`（默认关闭），未改历史C1/源码。参考原训练帧2/63/86/134/172，各用三个连续可见观测，以完整RGB颜色伴随梯度（非前20层）/完整中心/协方差有效性/原始双向光流选固定canonical行，随后同一批点全120帧验证。20候选建立16局部库，3831行计数去重3232；4候选只有9/10/13/14共同支持点，拒绝。后续176库—帧事件通过，71帧有部分支持，所有120帧平均接受点解释17.76%可见鸭子alpha；不是身份准确率或全模型正确表面比例。二维A0_F3仍有83后续可见片段帧，但固定点仅21帧联合通过；晚期新库不能冒称早期同一鸭子的材料点。补查源帧可见性后剔除1209点—帧事件，初版18.10%及代码/输出归档initial_target_only，固定库/数值门槛未变；同一片段源/目标双帧都需足迹/中心支持。终止/缺可见足迹/中心或流不符保持未知；跨参考行重叠仅记录假设，不合并物理ID。**局部对应已建立，完整物体跨遮挡身份/表面支持未验证，不恢复P1、不启动形态/接触训练。**

> 12测试、真实GPU完整贡献/直接点群合成/快照与完整状态/参数无改动通过。直接贡献积分相对误差≤约0.0168%；268冻结项与历史C1保护SHA核验，CPU全120中心光流与GPU最大差约5.73e-5px，全部固定点集/联合门槛/逐点未知/去重覆盖率重算。报告 `experiments/surface_correspondence/assessment.md`；`{plan,preflight,renderer_contracts,results,summary,independent_verification,manual_review}.json`。浏览 `output/surface_correspondence/index.html`；`support_timeline.png`、`center_footprint_3d.png`、20张六列中心/实际足迹对照；固定库banks/、完整surface_trajectories.npz与surface_assignments.npz。选点原始证据缓存 `runs/pidg_surface_correspondence/cache/` 必须保留供独立核验，不是训练检查点。初版人工看两图和九图，源帧支持修正后复看19/74/90/139/176/200六张最终图，非全部120/20。无新学习参数、Gaussian更新、外力/本构/学习式材料分割或自动跟进。下方旧训练命令仅供历史追溯。

> **最新：运动历史对应 R1 有边界测试及全时段滚动扩展完成（2026-11-02，零新 Gaussian 训练）**。独立 `variants/motion_correspondence` 为非神经二维位置/速度多假设关联，含可选历史外推、缺失协方差、合并拒绝、一对一、过期、持续歧义和完整 JSON 恢复；关闭接口精确委托旧 visible-FB。不是修改三维物理速度、不是物体/材料分割，不给未知 Gaussian 赋最近鸭子。12 测试通过，真实原帧42状态恢复后20帧逐项一致，只读 C1 final50k 完整 alpha 合成11帧。262冻结项和历史 C1 全部保护SHA通过。原型初版在隐藏反弹第二重现帧过早确认身份，源码/输出完整归档 `experiments/motion_correspondence/initial_probe`；已修正为未决身份锁存，8帧延长歧义及恢复通过，没有参数调优或新训练。

> 早期25帧的36个人工遮挡试验/72种子案例，平均预测误差9.5898→2.4890px；保守版本29符合可见流伪参照/0不符/43未知。用户要求继续测试后，先记录rolling_plan，再追加预定38个全时段窗口，19合格（早6/中6/晚7），19明确拒绝（10分量数不为四/9可见FB合并歧义）；76局部片段案例误差8.5617→2.0123px，保守44符合/0不符/32未知。局部窗口重新播种，不跨窗当全局身份；隐藏RGB/流不提供给追踪器。上述均为可见片段人工移除观测的伪参照评分，不能称自然完全遮挡或真实材料ID准确率。

> 全120训练帧自然候选接受数258/374/157（无历史/历史/保守，分母480），不是准确率；保守最终四种子全未知。完整合成中原早期局部点群对黄色鸭子贡献帧74仅12.95%，86及之后选帧为0，运动候选不能补回持久表面支持。**完整训练接入门槛未通过，不恢复P1、不加形态/接触项、不自动追加完整训练**。下一步需要独立身份线索/跨时刻表面证据与中心和足迹共同变形的持久表示；无法辨认的遮挡保留身份置换假设。报告 `experiments/motion_correspondence/assessment.md`，核验 `independent_verification.json` 与 `manual_review.json`，浏览 `output/motion_correspondence/index.html`，滚动对照 `rolling_review.png`、预测叠图 `forecast_examples.png`。人工检查图表及5个代表合成帧，非全部120帧。没有新自动跟进。下方所有旧训练入口仅作历史记录。

> **全时段表面支持审计已完成（2026-11-02，本轮无新训练）**：按用户最新要求，独立 output/surface_support 审计历史C1 final50k与拒绝P1完整16k，全部120训练帧的完整中心/实际足迹/全alpha分组合成/移除其他点遮挡/固定清晰足迹及相机，并以原始训练双向流做保守可见对应。旧CUDA诊断仅每像素前20层，覆盖alpha约62.3–87.6%，旧归一化贡献不能当完整贡献；未改旧源码/结果。本轮全合成仍确认C1从82起72帧、P1从86起70帧早期patch对可见鸭子为零，中心全在视野内但晚期离开黄色鸭子区域，去其他点遮挡亦不恢复。800宽2/74/86/200复核零支持结论。帧86局部原未知80点分别承担48.88%/51.25%贡献，只作该帧探针，不赋予物体ID。120个原mask全是0/255；42→43两个图像分量合并，后续重识别未建立，保持未知。**不能据此证明真实材料漂移，更不能恢复P1或直接加接触约束**；应先取得可靠object/material身份，或具有持久物体/表面坐标且中心和足迹共同变形的表示，再重新审计。没有新候选/训练/自动跟进。报告 experiments/surface_support/assessment.md、representation_requirements.md；全序列浏览 output/surface_support/index.html，中心叠图center_review.png，3D轨迹center_trajectories_3d.png。9测试、只读GPU契约、完整状态/快照/图/数据/流SHA及历史C1保护通过；人工审阅两模型各8帧及图表，未冒称全240图人工审阅。独立验证 independent_verification.json，manual_review.json。全部冻结源码与旧模型保留。

> **19:11 当前终态：P1在16k覆盖审阅门槛未通过，已停止剩余34k；本轮不是已训练完50k，也不是已验证形态损失无效。** 原完整3k→16k新增13k，尚无一次新形态项求值/更新。图991个部分标记点/646邻域，早期可见鸭子像素贡献64.60–67.31%，到帧74降至7.29%，86/99/108/146/200为零；已人工看五帧分组与十个早中晚训练帧。图的推断标签隔离通过，但当前监督没有被证实覆盖用户要修复的中段可见形态，因此不盲目训练，不自动追加候选。可见贡献不足不能区分视角/遮挡/有效足迹或对应漂移，不能断言材料ID错或缺外力。历史C1及所有旧模型未变。

终态 `runs/pidg_shape_coherence/controller.json` 为 `rejected_graph_gate`，`candidate_graph_review.json` approved=false/manual_review_pending=false，**不要恢复该候选或运行旧协调器**。16k完整checkpoint SHA b36bfde8e25ffe1057e493a4bcbb40c17236882151e07118f95d091048a82b17，参考图SHA ff302f2d5c0a4853ae57159a43e5b104be680a77c0a6e1098dffa76b76715b79，已核验源码/数据/共同起点/sidecar/有限模型和优化器、历史控制全部SHA。只修了output审阅脚本：原帧98不是训练帧，按训练元数据改用99，未访问本候选测试。报告 `experiments/shape_coherence/assessment.md`，凭据 `{candidate_graph_evidence,candidate_graph_review,gate_verification}.json`，十帧浏览 `output/shape_coherence/index.html`、总览gate_review.png；没有新的50k测试/视频/效果比较。下一步应先验证全时段物体/表面支持、中心与实际渲染足迹的一致性，再决定结构方案；这些尚未实施。自动跟进 **c1已成功删除**。下方18:46启动记录仅保留追溯，不能据此继续训练。

> **18:46 最新用户授权：处理多鸭子接触时形态损坏。独立 P1_local_shape 已实现并通过12测试、真实C1状态形态梯度/优化器更新、GPU100步/70→100完整恢复、flags-off20（启动修复后重复一次，实际关闭预算40）。历史C1原样复用，不重训，不开光流，不引入外力/本构/新参数。新目录 variants/shape_coherence，协议/诊断 experiments/shape_coherence/{plan,diagnosis,preflight,source_freeze}.json 和 diagnosis.md。冻结SHA214b07fc42532c912201dca310387be2f05a85091af1ada4f12fe79ec26390f8。首个启动错误仅缺诊断路径迁移凭据，未发生更新；已修复并归档旧冻结凭据。**

代码/预检里程碑 **a48a790c**。唯一新协调器会话 **37608**、PID **235736**，训练PID **225988**；启动记录 `experiments/shape_coherence/launch.json`。本轮已创建每10分钟自动跟进 **c1**，这是删除旧光流自动化后新建的形态任务，不能按旧光流指令执行。常规健康推进保持安静；完成交付或图门槛失败形成完整诊断交付后删除。

**本轮不是已完成训练。** 唯一候选从原完整3k SHA6da523ea247b4068ec9bd6a7c8f83dacb7506c698daaeecc9eb0ecd93ea1ebce至50k，原稠密化15k结束后16k构建参考图，自动保存暂停、等待本聊天人工审阅。启动/恢复入口：激活enable_cuda.ps1后，`envs/pidg-cu128/python.exe -u output/shape_coherence/continue_run.py --hours 12`。只读状态 `output/shape_coherence/status.py`。不要重复启动；健康训练不占GPU，不改冻结源码/配置/数据。若退出，先核验checkpoint SHA/保存边界，归档移出超界shape_loss等诊断再恢复。不要使用旧C1/O0/O1等协调器。

16k停止后运行 `output/shape_coherence/review_graph.py`，实际从完整状态加载，核验图SHA/同组连边，渲染训练早期及62/74/86/98/108等中段，人工看 `shape_reference/early_training_components.png` 与 `output/shape_coherence/candidate_graph/`。必须把中段实际贡献列入判断，再写 `experiments/shape_coherence/candidate_graph_review.json`，`approved` 和 `graph_sha256` 绑定 `run/shape_reference/reference_graph.pt`。不自动批准。早期物体内高置信参考点只是局部表面patch，不是真实持久材料ID；历史C1审计显示它们中后段几乎不再参与鸭子渲染，不能把局部形态项说成完整物体跟踪，也不能用小残差冒称修复碰撞。若无可靠覆盖，记录未过门槛，不盲目训练；需要结构修正时先说明证据，保留这一候选的预算边界。

图通过后用相同入口接续至50k，输出辅助脚本将锁定验证选择，复用C1既定47k best/50k final，仅渲染新P1，生成GT|Render PNG/两视频/ZIP与验证接触ROI、V2独立二维运动对照。`output/shape_coherence/finalize.py`/`verify_results.py` 是允许修复的后处理辅助脚本；不要改锁或重训已完成模型。预先门槛：验证PSNR下降≤.2dB、LPIPS增幅≤5%、验证帧62..108固定黄色鸭子ROI MAE降低≥10%，人工无新增撕裂/粘连/塌缩；窗口是诊断范围，不等于独立接触GT。需核验快照/完整状态/图/输入SHA、锁定顺序、拼接像素/PSNR/视频帧/ZIP，人工看早中晚及原帧85、视频解码图，补assessment并独立结果提交。原C1没有逐步相机日志，新P1仅每100步记录形态诊断，不冒称逐步实际采样一致。原PIDG最后一步无optimizer.step，GPU100为100次求值99次更新。仅一个完整候选，无额外候选、外力或材料未知量。交付后删除本轮自动跟进（若已创建）；未创建时不能冒称后台跟进已安排。

> **C1光流比较最终完成（2026-11-02 17:34后核验）：历史C1未重训且全部受保护SHA未变；仅O1从同一原完整3k到50k，已锁定best=50k（历史best=47k）。验证PSNR39.683393→39.312937（−0.3705dB），测试验证best40.334404→39.900620（−0.4338dB）；独立验证速度推进EPE降低16.43%（26/30对改善），几何EPE恶化0.84%。预先综合收益false，保留C1画质基线，不追加候选。源码/输入/完整检查点/快照/共同起点SHA、47000次采样、470事件/468求值/467更新、200拼接PNG像素及独立PSNR、4×50视频解码帧、ZIP逐成员字节/CRC全部通过。早中晚、原帧85全尺寸、全部best视频解码序列和图表人工审阅完成；不冒称实时播放审阅或3D物理验证。结果 `experiments/c1_flow_ablation/{results_reuse,final_verification_reuse,independent_verification_reuse,review_statistics}.json`，评估 `assessment.md`，同步页 `output/c1_flow_ablation/index.html`。旧O0继续取消/保留/排除，禁止重启旧main。本轮完成，交付时删除c1自动跟进。**

以下16:12记录是已结束队列的恢复历史，不应再启动。


更新时间：2026-11-02 16:12 UTC（Europe/London）。
工作目录：`F:\A_Edinburgh\PhysGauss`；Windows / PowerShell。

> **16:12 用户纠正，现行唯一方案：原模型不重新训练，关闭光流组直接复用历史 `runs/pidg_color_repair/C1_repaired/rubber_duck`。旧O0重训已保存停止在9712步（SHA525c29cb286c681f2ea0872951b22fdac03fc6f18a01a01d458716d26f2b08c1），保留全部记录但排除结果，禁止恢复O0。旧双组协调器会话88306已正常退出，旧根目录stop_queue.request故意保留以阻止误启动旧main。只训练O1_C1_flow_on，仍从完全相同原始3k SHA6da523...ebce到总50k，开光流以外的模型/损失/时钟/动量均不改。**

新单模型协调器入口 **`output/c1_flow_ablation/reuse_control.py`**，会话 **33935**，日志 `experiments/c1_flow_ablation/reuse_control_coordinator.log`；新队列控制文件在 `runs/pidg_c1_flow_ablation/reuse_control_queue/`（controller/active/lock），O1模型目录不变。状态查询仍为 `envs/pidg-cu128/python.exe output/c1_flow_ablation/status.py`，已改为显示历史控制复用和新队列。禁止运行 `variants.c1_flow_ablation.run` 原双组main，也不移除旧根目录stop_queue.request；新队列独立lock，不受旧队列stop影响。恢复命令：激活enable_cuda.ps1后，`envs/pidg-cu128/python.exe -u output/c1_flow_ablation/reuse_control.py --hours 16`，先核验新进程确已退出、最后完整checkpoint SHA/边界，归档超界诊断再接续。健康推进不另占GPU，不修改冻结训练源码/配置/数据。

`experiments/c1_flow_ablation/reuse_control_plan.json` 是用户更正后的协议，旧plan/source_freeze作为历史凭据保留不覆盖。原16测试/100步GPU短训/70→100恢复/flags-off20/实际flow-only更新预检仍适用于同一个不变的O1训练代码；新prepare-only检查已核验原控制完整checkpoint、best/final快照、原有PNG/指标/选择锁SHA、同一3k来源及相同模型配置。新增控制复用只读辅助脚本 `output/c1_flow_ablation/compare_reuse.py`，不重新训练或重新选择历史控制。历史C1验证最佳为47k（验证PSNR39.683393，测试best40.334404）；最终50k测试40.372128，不能把50k分数冒称验证best。

O1完成后新队列锁定历史既定47k选择和O1验证best，再只渲染O1 best/final；历史C1直接使用已有PNG/指标导出。辅助脚本输出新旧GT|Render PNG/两视频/ZIP和同步页，固定V2验证raw流、完整状态/快照SHA/原控制未改/采样和光流次数/像素PSNR/帧数/ZIP检查。结果 `results_reuse.json`、`motion_validation_reuse.json`、`final_verification_reuse.json`；人工查看早中晚、原帧85全尺寸、png_review和decoded_video_review及视频，补充assessment和人工核验状态，提交结果里程碑后交付。原历史没有逐步camera_sampling日志，不冒称核验两条实际采样日程相同；相同3k字节状态/配置可核验。只有这一组新完整训练，不增加候选、不加外力/本构/材料未知量。自动跟进仍为c1，已改为复用控制方案，完成交付后删除。

以下16:00原双组安排已被用户纠正，仅作追溯，不能据此恢复O0。

> **16:00 最新授权／正在执行：用户要求用最初C系列表现最好的模型做只开／关光流的完整对照。按历史验证选择C1_repaired（验证PSNR39.664895，测试40.372128仅参考）。新增独立 `variants/c1_flow_ablation`；代码/预检里程碑558019e3。两组新训练 O0_C1_flow_off / O1_C1_flow_on 从完全相同原始C1完整3k状态SHA `6da523ea247b4068ec9bd6a7c8f83dacb7506c698daaeecc9eb0ecd93ea1ebce` 到总50k，各新增47k；不用历史50k权重继续，也不用历史C1代替新控制。唯一配置差别是 `observed_flow_enabled`。保留C1原独立velocity/stress、原Cauchy动量实现、frame-index共享时间、RGB/SSIM/LPIPS/alpha/前景损失及优化器/稠密化；不混入L1/M1/N1的共享应力、guard或新动量公式。开光流采用已修正相机/offset/贡献权重的raw训练UniMatch缓存，无H3深度；几何流和独立速度推进流各半，权重.01、每100步、3k后2k ramp；只改这一个光流监督包。**

预检：16测试通过；两组100步GPU短训，O1完整70→100恢复，all-flags-off20，真实flow-only优化器更新（grid/deformation/velocity_head均改变，私有stress_head无直接光流梯度），继承C1图像loss/时钟精确一致，真实共同3k字节SHA与模型配置一致；无新增参数/优化器组、无额外前缀拟合。协议和冻结凭据 `experiments/c1_flow_ablation/{plan,preflight,source_freeze,common_start_checks,prefix_audit,optimizer_update}.json`。source_freeze SHA `fe72d2a5a15b53599df60af6dfee63f0fab186b13adbc662811649fc9786f56e`。

唯一完整训练协调器会话 **88306**、PID **231636**，O0初始训练PID **237932**；日志 `experiments/c1_flow_ablation/training_coordinator.log`，启动凭据 `launch.json`，先O0再O1，禁止重复启动或另占健康GPU。命令：先 `. ./reproduction/scripts/enable_cuda.ps1`，再 `envs/pidg-cu128/python.exe -u -m variants.c1_flow_ablation.run --hours 16`。当前本聊天自动接续 **c1**（每10分钟）；此前N1的automation已删，不能混淆。状态只读：`envs/pidg-cu128/python.exe output/c1_flow_ablation/status.py`，再核对实际进程及日志更新；健康推进保持安静，不改冻结源码/配置/数据。

若意外退出未完成：先核验最后完整checkpoint SHA/保存边界，归档并移出超界 `camera_sampling.jsonl`、`observed_flow.jsonl`及其他诊断，再安全续跑；不要把未保存步骤日志算到新训练。两组均完成后才锁定验证PSNR选择，再打开50测试帧；已有selection_lock必须原样保留。协调器自动导出best/final的GT|Render PNG/两视频/ZIP，复用冻结V2验证观测做独立运动诊断，并核验源数据SHA、完整状态/快照、采样日程、像素、独立PSNR、视频帧数、ZIP。真实后处理问题用独立output辅助脚本修复，不重训、不改锁、不改冻结训练源码。`variants/c1_flow_ablation/results.py`生成结果和初步assessment，但completed后仍需人工查看早中晚、原帧85全尺寸PNG、`png_review.png`和`decoded_video_review.png`及视频，补全assessment和final_verification的manual_review_pending状态，再提交独立结果里程碑并交付 `output/c1_flow_ablation/index.html` 和两组视频/报告。

判断以新O1−新O0为准；预先综合门槛为验证PSNR下降≤.2dB、LPIPS增幅≤5%、验证速度推进EPE降低≥10%、几何EPE增幅≤5%，各轴如实报告。不得把估计光流或内部一致性当3D物理真值；单场景单种子、CUDA非确定性、历史测试暴露仍适用。只有这两个完整模型，不自动新增候选，不加入外力/本构/材料未知量；保留历史模型和无关改动。最终核验交付后删除c1自动跟进。

以下15:02及更早为已结束实验的历史记录，不代表本次C1光流消融结束。

> **15:02 N1最终交付：已完成总50k、验证选择锁定（best=final50k）、测试/GT|Render PNG/两段视频/ZIP、独立核验、同材料点双算子比较和三维图审阅。本轮已结束，自动跟进 `automation` 已删除，不按以下历史命令重训。相对M1：测试PSNR39.701836→39.909519（+0.2077dB），验证PSNR仅+0.0105dB，测试LPIPS略恶化1.26%；独立二维速度推进EPE降低3.85%、完整轨迹EPE降低5.11%。共同64锚点在帧2/83/155上，原算子Huber中位数降低2.76%，新算子降低7.21%，均未达到预先25%门槛，因此综合收益false。帧83异常明显减轻，帧2反而恶化，不能泛化为物理成功。**

最终报告 `experiments/coherent_momentum/assessment.md`；同步新旧视频/逐帧比较页 `output/coherent_momentum/index.html`；三维全64点位移图 `output/coherent_momentum/rollout_displacement_3d.png`。独立 `final_verification.json` passed且manual_review_pending=false：源码/数据/共同3k/完整50k/快照SHA、验证选择时序、468光流/420动量评估（419次更新）、100张拼接图逐像素、独立PSNR最大误差5.3e-6dB、两视频各50帧/5秒、ZIP CRC均通过；全部参数及69个优化器状态张量有限。完整50k SHA `36308753137b7c1d796ad95f337b50c61aa2d71744dfc32b15e8aecf0accc84d`。best与final模型权重相同，视频标题不同。

剩余问题及边界：率修正散度仍主导；N1时间步减半的加速度变化18.8%～22.1%，空间邻居变化也明显；保持M1 guard使共享物理梯度受到强限制。详见 `field_sensitivity.json` / `additional_verification.json`。N1仍是同一轨迹加软物理目标，不是物理积分器唯一生成渲染中心，不能保证严格共轨；未加外力/本构/未知材料，也不把代数零偏差或h²缩放残差当独立真值。单场景单种子及历史测试暴露限制结论。不再自动增加完整候选；全部历史模型及无关改动保留。

以下13:43及更早记录均为历史追溯，不表示仍有训练或跟进在运行。

> **13:43 新授权：用户要求统一物理动量与形变运动并与之前模型比较。已明确L1/M1本来就由完整渲染中心轨迹导出速度/加速度，因此本轮不是首次消除独立速度分支。新增独立开关候选 `variants/coherent_momentum` / N1_coherent_momentum：统一h=1/199区间的轨迹/弦速度/加速度及物理推进位置，保留stress_pred、原应变率修正和M1梯度策略；用32邻居、FP64归一化二次局部拟合直接计算应力Jacobian与速度Hessian，替换原连续两次线性拟合。不新增参数、外力或本构；这是多个数值环节的改动，不冒称单标量消融或精确物理求解器。24项测试、100步GPU短训70→100完整恢复、20步flags-off、实际优化器更新/私有梯度精确保留/共享梯度上限全通过。`experiments/coherent_momentum/{plan,diagnosis,preflight,source_freeze}.json` 为冻结协议；共同3k检查点SHA仍为 e361f4ba241a77e4158d341986bf86f2797e24c9a560817b04ad6c18b3a4caa5，张量/优化器/RNG字节相同。仅这一个完整候选，从3k到总50k，比较历史M1，禁止重训旧模型。**

本轮命令：`. ./reproduction/scripts/enable_cuda.ps1` 后 `envs/pidg-cu128/python.exe -u -m variants.coherent_momentum.run --hours 12`。唯一协调器日志 `experiments/coherent_momentum/training_coordinator.log`；状态 `envs/pidg-cu128/python.exe output/coherent_momentum/status.py`。健康训练不另占GPU；冻结后不改源码/配置/数据。完成验证选择锁定后自动测试best/final并导出GT|Render/视频/ZIP，使用同一冻结V2观测与同一批canonical锚点、训练时刻，对新旧两个模型分别应用新旧两种动量算子。标准已预先写入plan，不能只用新算子残差降低宣称物理更可靠；推进误差只是h²缩放残差，不是独立验证。还须运行 `output/coherent_momentum/verify_results.py`、人工审阅早中晚/原帧85和视频、补充assessment并提交独立结果里程碑。若意外退出，先核验完整保存边界、归档超界诊断（含coherent_momentum和gradient_guard JSONL）再安全续跑；已有selection_lock不得重写。之前“本轮结束”的13:26内容仅指M1旧轮。

启动凭据：唯一完整协调器工具会话 **90314**，PID **228112**，训练PID **228568**，代码/预检提交 **b8fbbda8**。已实际恢复3k并推进至4k以后。已在本聊天创建每10分钟自动接续，当前ID **automation**（新创建，名称“统一轨迹动量实验接续”；早期同名ID的删除记录为历史）。常规推进保持安静，只报告结论/失败/完成/需要输入；交付核验后删除。`output/coherent_momentum/rollout_alignment.py` 将在GPU空闲后生成同材料点、同一算子下的M1/N1三维物理推进与渲染轨迹图，明确其不是物理GT。

> **13:26 最终交付：M1_guarded_shared 已完成总50k、best49k/final50k测试、GT|Render导出、独立核验和人工图像/视频审阅。本轮结束，禁止按下文历史启动记录重训；自动跟进 `pidg` 已删除。测试best PSNR由L1的33.965832升至39.701836 dB，LPIPS约降低50.3%；验证二维速度推进EPE降低8.99%，未达到预先10%综合收益门槛，因此只确认重建恢复，不确认物理预测更可靠。未缩放动量残差恶化；同一组共同3k材料点诊断表明，原应变率修正项散度是大残差的主要数值贡献。加速度差分对步长仍敏感，不据此归因缺外力。不启动更多完整训练。单场景单种子，首次guard生效前已有0.26–0.65 dB差异，不能把全部收益唯一归因guard。详细证据、限制和下一步只读/解析测试建议见 `experiments/momentum_repair/assessment.md`；新旧同步视频与逐帧对照 `output/momentum_repair/index.html`。**

最终核验：`experiments/momentum_repair/final_verification.json` passed，人工审阅已完成；100张拼接PNG逐像素一致，独立PSNR误差小于4.3e-6 dB，两段视频各50帧，ZIP CRC和SHA通过。源码/数据/共同3k/完整50k/渲染快照均核验；468次光流、420次动量评估和419次实际动量更新（原trainer末步不执行optimizer.step）。完整50k SHA `c54c4bf8b53f9129e8297e7f7661cdc8e839ad35fba0dbff2ef9be775a0f5684`。原协调器仅在并行导出mkdir处失败；`finish_report.py` 串行建目录后接续，保留原selection_lock SHA `d2617e73482d38d30117da5c08c440f4d53017297709c91c599e0e05b92a660d`，没有重训、重选模型或修改冻结训练源码。所有旧模型和无关改动保留。

以下11:56及更早内容仅为历史追溯，不代表仍有训练或自动跟进。

> **11:56 最新授权与接续：用户要求“找到问题自动进行改进”。已完成三个真实同批训练帧的梯度/差分诊断：L1已加权动量/RGB梯度在Grid4D约4884–15748倍、形变网络约850–3241倍；二阶差分未稳定收敛，尚不能据此证明整个退化的因果或缺外力。新增独立开关变体 `variants/momentum_guard`，只对动量共享形变梯度施加同批RGB梯度10%上限，私有stress头及已有force_sigma梯度保留；这是显式一阶梯度策略，真实前向物理残差不缩放。没有新参数、没有新外力/本构/材料未知量，网络/步长/原权重不变。15项测试、100步GPU短训/70→100完整恢复、20步flags-off、实际优化器更新、私有梯度精确相等和共享上限验证全通过。里程碑 `1af20b2`。唯一完整训练协调器工具会话21588，controller PID237464，训练PID235388，日志 `experiments/momentum_repair/training_coordinator.log`；从与L1相同扩展完整3k SHA e361f4ba241a77e4158d341986bf86f2797e24c9a560817b04ad6c18b3a4caa5 到总50k。不要重复启动或修改冻结源码/配置/数据。先读 `experiments/momentum_repair/{diagnosis.md,plan.json,preflight.json}`，执行 `envs/pidg-cu128/python.exe output/momentum_repair/status.py`。健康推进保持安静，不另占GPU。结束自动验证选择锁定→best/final测试→GT|Render PNG/视频/ZIP→V2独立运动与未缩放动量残差→report。completed后仍须运行 `output/momentum_repair/verify_results.py`，人工查看生成的早中晚/原帧85和视频审阅图及图表，补充assessment并独立结果提交。比较M1−L1，报告图像质量、独立二维运动、内部结构一致性/原始残差，不能把小梯度或定义上零速度差说成物理真值。单场景单种子、历史测试暴露。仅新增这一个完整候选，不无边界扩训。完成交付后删除新自动跟进。旧L0/L1及以下历史段落不可作为重新训练依据。**

安全续跑：先确认协调器/训练进程退出，核验最后完整检查点与SHA，归档并移出超出保存边界的诊断（包括momentum_gradient_guard.jsonl），再 `. ./reproduction/scripts/enable_cuda.ps1; envs/pidg-cu128/python.exe -u -m variants.momentum_guard.run --hours 12`。若已到50k且已有selection_lock，禁止重写选择锁；直接在独立辅助脚本中接续未完成render/export/report，保留原锁和原始时间凭据。诊断导入前必须设PIDG_HASH_SOURCE_DIR/BUILD_DIR并先导入本轮扩展架构，避免旧root陈旧build锁和错误网络类。保留所有历史模型与无关脏文件。

已创建每10分钟本聊天自动接续，ID `pidg`；正常推进不通知，完成核验/交付后删除。启动凭据 `experiments/momentum_repair/launch.json`。旧ID `automation` 已在上一轮交付时删除，不能误认成当前跟进。

> 给接手 Codex：K1/K2及后续L0/L1均已完成训练、导出、运动诊断与结果核验。本轮结构尝试已结束，先读下方最新记录及 `experiments/unified_flow/assessment.md`。后面的历史进度和恢复命令只供追溯，禁止据此重启或重复训练。

## 1. 用户目标与授权

用户最新研究要求：先加入运动光流训练一组，再以“速度从形变轨迹导出”的结构做另一组比较。此前还要求完整训练后输出 Ground Truth 与 Render 的拼接结果。

用户现在要求生成本交接文件，方便另一位 Codex 接手。**没有要求停止训练**；原有队列继续运行。两组完整训练、评估、导出均已授权，无需再次询问是否继续。不要自行加入外力、本构或其他未知量联合训练。

用户中文沟通。不要把“已启动”写成“已完成”，不要用内部自一致性代替物理真值验证。未获授权不要发邮件/Slack 消息；不要主动创建新聊天或子代理。

## 2. 接手时的运行状态

**08:40 UTC 最新状态：L0/L1均完成总50k、全部GT|Render导出、V2运动诊断及独立核验。L0验证选择49k，L1选择50k；验证PSNR33.735493／33.755262，测试best PSNR33.953815／33.965832。L1验证LPIPS降低6.00%，速度推进EPE仅降低0.12%，轨迹EPE增加0.34%，不满足预先综合收益门槛。原帧85两组都仍漏悬空鸭子。详见 experiments/unified_flow/assessment.md。诊断导入时hash环境变量设置过晚，原协调器23682/PID227560空等陈旧root构建锁；已停止该已完成训练的协调器，以独立finish_report.py提前设路径，08:36:14完成，不改冻结源码、不动旧锁、不重训不重渲染。恢复会话22702及CPU核验82770均成功退出。final_verification.json与snapshot_verification.json通过；早中晚、原帧85、视频解码和图表均人工检查。本轮结构尝试已完成，不启动更多完整训练；结果里程碑提交及自动跟进删除将在本次交付完成。**

**05:45 UTC 最新状态：下一轮完整训练已启动，唯一协调器工具会话23682，日志 experiments/unified_flow/training_coordinator.log。L0_corrected_trajectory 正从扩展共同3k状态恢复，L1_shared_stress 随后自动训练；两组各总50k。不要重复启动。`envs/pidg-cu128/python.exe output/unified_flow/status.py` 查看进度。新变体/协议/预检里程碑提交 e058831；上一轮完整结果和几何审计提交495dff9。15项测试、两组100步GPU短训、70→100恢复、20步flags-off、实际隔离优化器更新均通过。实际更新：L0旧应力头改变/新头不变；L1新应力头改变/旧物理网不变；两组grid/形变均改变、旧velocity头不变。完整共同起点继承张量/优化器状态/RNG逐项核验通过，凭据 experiments/unified_flow/{preflight,optimizer_update_checks,prefix_extension_checks}.json。原3k SHA6da523...ebce，扩展3k SHA e361f4ba241a77e4158d341986bf86f2797e24c9a560817b04ad6c18b3a4caa5；只增加两组同样的seed20261102、std.001、bias0应力Linear(256,6)及空状态优化器组，没有额外拟合。**

新路径：`variants/unified_flow/`（源码已冻结）、`experiments/unified_flow/plan.json`、`runs/pidg_unified_flow/`、`data/derived/unified_flow_train_only/`、`output/unified_flow/`。两组都用正确raw source→target相机流、不使用H3深度、修正CUDA offset/合成权重，真实加载器相机约定不改。L0保留独立PIDG stress；L1在形变Grid4D/同一latent上输出stress6，速度两组都来自完整轨迹。动量/光流权重和差分步长不同时调；共享表示未必解决过大的二阶差分梯度，这是受控试验。相对旧K2还改变流程，不能把K2→L1收益算作共享分支收益。新协调器会训练两组→验证选择锁定→测试best/final→像素/PSNR/视频/ZIP导出→复用冻结V2验证观测做运动诊断→检查闲置权重/有限参数→报告→completed。完成后仍需人工审阅PNG、图表、完整检查点SHA和选择时间，补充原因分析并提交结果。若需恢复，确认旧进程退出、归档超过完整检查点的诊断，然后 `. ./reproduction/scripts/enable_cuda.ps1; envs/pidg-cu128/python.exe -u -m variants.unified_flow.run --hours 12`。禁止编辑已冻结的新变体、协议、父级源码或测试。旧 K1/K2 已交付，勿重跑旧后处理。

**04:25 UTC 最新状态：K1/K2 全部训练、导出和三阶段后处理完成。测试 PSNR 38.301304 / 34.206497。进一步发现原辅助相机 FoV 与真实加载器不符，以及 CUDA mean-minus-pixel 偏移符号、metadata 写入后 T 权重约定有误，均用实际源码及 CUDA 单 Gaussian 测试核验。原训练源码/模型/缓存/V1诊断保持冻结，不能把本轮退化唯一归于结构。新增 `evaluate_motion_v2.py`、`gradient_balance_v2.py`、`summarize_results.py` 已实际执行；`motion_validation_v2.json`、`gradient_balance_v2.json`、`diagnostic_audit_v2.json` 和 assessment.md 为当前解释依据，旧 motion_validation.json / gradient_balance.json / motion_comparison.png 已被 V2 替代。真实相机下 K1/K2 覆盖率99%以上；速度推进 EPE 1.193206 / 0.846755 px，但几何 EPE 0.783709 / 0.846833，K2 仍不满足综合门槛。5项新增测试（含真实 loader CameraInfo 和 CUDA offset/权重）通过，日志 output/flow_comparison/geometry_v2_tests.log。已人工看原帧85拼接、解码视频0/21/49图及训练/运动图。下一轮应使用统一修正流程的分离应力对照 L0 和共享应力候选 L1，以隔离流程修复与共享分支。几何修正与相机契约代码已在 `variants/unified_flow/{flow,camera_contract}.py`；尚未整合完整训练器/启动下一轮。下一步继续实施下述候选，不能重复跑旧后处理。**

**04:08 UTC 接续记录：两组均已完成 50k，验证选择锁已写入。K1 验证 PSNR 38.026127，K2 33.848821；尚不能用内部零速度偏差替代独立运动诊断。原队列正在顺序渲染/导出。已启动唯一后处理等待器 `output/flow_comparison/postprocess_queue.py`（工具会话 87816）：等待 `progress.json` 为 completed 后，顺序执行 evaluate_motion、gradient_balance、finalize。先检查 `experiments/flow_comparison/postprocess_status.json` 和各阶段同名 `.log`；等待器/后处理仍活跃时禁止重复运行 GPU 诊断。若 failed，先修复该阶段并接续，不能冒称已通过。完成之后仍需人工查看输出、补充报告、结果提交和按用户条件做下一轮结构实验。**

最近核验：2026-11-02 02:57 UTC。用户已授权在比较后继续改进，详见 `experiments/flow_comparison/followup_plan.md`。

| 项目 | 状态 |
|---|---|
| K1_flow_independent | 已完成 50,000 / 50,000 步 |
| K1 最近/最佳验证 | 第 50,000 步，**验证 PSNR 38.026127 dB**，LPIPS 0.01371659 |
| K1 内部速度相对误差 | 五个训练时间探针平均约 **99.80%**，不是速度真值误差 |
| K2_flow_trajectory | 正在训练，约 4,100 / 50,000 步；4k 验证 PSNR 24.92583 dB |
| 整个队列 | `training`；最近检查没有数值错误 |
| 数据/单元/GPU/短训预检 | 全部通过，见第 5 节 |

上述进度会过时，以以下文件为准：

- `experiments/flow_comparison/progress.json`
- `experiments/flow_comparison/monitor.html`
- `experiments/flow_comparison/training_coordinator.log`
- `runs/pidg_flow_comparison/<模型>/rubber_duck/train.log`

**检查进度（不占用 GPU）：**

```powershell
Set-Location F:\A_Edinburgh\PhysGauss
envs/pidg-cu128/python.exe output/flow_comparison/status.py
Get-Content experiments/flow_comparison/training_coordinator.log -Tail 20
```

旧协调器会话 **14891 已退出**，最后未保存日志到 41,760 步。接手时已确认没有 Python 进程，核验 41,000 步完整检查点，归档并移出超过该边界的 7 条/文件未保存诊断记录，再于约 02:40 UTC 恢复。新协调器工具会话 **76990**，日志 `experiments/flow_comparison/resume_coordinator_20261102.log`，参数 `--hours 12`。K1 随后完成，K2 已启动。不要假设另一聊天可以访问该会话，优先根据日志时间戳和进程判断。

恢复凭据：`experiments/flow_comparison/resume_reconciliation.json`；完整原日志归档 `output/flow_comparison/resume_archives/20261102_024041/`。未改权重、配置或冻结源码；动量 KDTree 缓存未序列化、恢复时重建，因此不声称与不中断运行逐位一致。若再次中断，按新的检查点边界重新归档日志；`reconcile_resume.py` 是本次 K1/41k 的一次性脚本，不可原样重复用于 K2。

已建立当前聊天的自动接续检查（automation ID `automation`，每 10 分钟）。训练正常推进时保持安静；结束后继续诊断和用户授权的条件式下一轮。不要重复创建跟进任务，也不要重复启动协调器。此次接续记录/辅助脚本提交见 `git log`。

**只有确认旧协调器和训练进程已经退出后，才执行恢复命令：**

```powershell
Set-Location F:\A_Edinburgh\PhysGauss
. ./reproduction/scripts/enable_cuda.ps1
envs/pidg-cu128/python.exe -u -m variants.flow_comparison.run --hours 12
```

该命令会核验冻结源码、复用已通过预检、恢复已有完整检查点，并跳过已完成工作。不要直接裸跑上游 `train_pidg.py`，否则会遗漏变体配置/完整恢复约定。不要只因短时间没有新验证结果就判定进程已停：验证和写检查点会暂时停留在整千步。

## 3. 已固定的实验设计

完整协议：`experiments/flow_comparison/plan.json`。

- 场景 Rubber Duck；seed0；**120 train / 30 validation / 50 test**。
- 两组复用**同一个完整 3,000 步检查点**，含模型、优化器、相机栈与 RNG，继续到**总计 50,000 步**，不是额外再训 50,000 步。
- 起点：`runs/pidg_color_repair/shared_prefix/rubber_duck/checkpoints/resume.pt`。
- 起点 SHA256：`6da523ea247b4068ec9bd6a7c8f83dacb7506c698daaeecc9eb0ecd93ea1ebce`；已核对该目录 `resume.json`。
- K1：`velocity_source="independent"`，独立速度头 + 光流。
- K2：`velocity_source="trajectory"`，完整渲染中心轨迹的有限差分速度 + 相同光流。保留参数对象、state_dict 键和优化器组，冻结未使用的独立速度头；**应力头继续训练**。
- 两组配置只应相差 `velocity_source`；相同固定单调物理时钟 `tanh(2*t-1)`，相同图像、覆盖、邻边应变损失、相同光流和局部动量算子。
- 两组都不使用额外软运动学一致性损失：`kinematic_weight=0`。
- 光流权重 .01，每 100 步，3k 后启用、2k 渐增；RGB 1600×900，光流与对应关系 400×225。
- 动量权重 .01，每 100 步，8k 后启用、4k 渐增；64 anchors，12 neighbors，Huber delta=1，rho=1。
- 速度有限差分 h=.001；加速度 h=1/199；边界使用二阶单侧差分。
- 两组全部训练结束后，先锁定验证集 PSNR 最佳快照，再打开最终测试渲染。导出 `best` 和 `50000` 两种快照。

**关键解释限制：**为支持轨迹速度且避免 CUDA hash encoder 不支持的坐标 Hessian，两组共用了新的局部 Lagrangian 动量离散化。相对于历史 H3/J1，还改变了光流、时钟或动量算子/分阶段设置；**不能把历史模型到 K1 的差异全部归因于光流**。这是已向用户说明的设计。K1/K2 之间的主要受控差异是速度来源。

## 4. 物理与代码含义

共同帧时间轴已修复为 `t=(frame_id-1)/199`。当前问题不是再次发生 train/test 错帧，而是两个独立函数在同一 X、同一 t 给出的 `v_pred` 与 `∂t x` 不一致。

- 形变场：四个 3D hash grid（XYZ/XYT/YZT/XZT）与形变网络。
- 原物理场：另一套六个 2D planes + ID embedding + MLP，输出 velocity3 与 stress6；**不是两个分支共用同一套 grid**。
- 必须使用完整渲染中心 `x(X,t)=R(q(X,t))X+b(X,t)`，不能只对平移 b 求导。
- K2 速度从该中心轨迹得到。其内部速度误差为零是**结构定义**，不是物理准确性证据。
- 新局部动量项在固定材料标签 X 上求加速度，在形变后世界位置 x 上用局部加权拟合求速度梯度/应力散度；没有额外 Eulerian 对流项。
- 保留 PIDG 的预测应力和原速度梯度偏应力修正；没有加入 Neo-Hookean、外力、自由刚度或材料分割。
- 时间单位仍为归一化场景时间，不能直接称米/秒。
- 原训练器控制台 K2 的 `velocity mean` 仍打印未使用的旧速度头均值；以带来源标签的运动学/光流/动量日志为准。

主代码：`variants/flow_comparison/`。

| 文件 | 作用 |
|---|---|
| `run.py` | 顺序训练 K1→K2、完整恢复、选择锁定、测试渲染、导出 |
| `prepare.py`, `code/` | 独立冻结 fork；原 PIDG 与旧变体未改 |
| `model.py` | 时钟/速度来源开关，冻结闲置参数，保留组布局 |
| `temporal.py` | 完整中心轨迹、检查点化查询、速度/加速度差分 |
| `local_momentum.py` | 当前坐标局部梯度、应力与材料坐标动量残差 |
| `flow.py` | 仿射 Gaussian 光流、Torch 投影/协方差梯度 |
| `geometry.py` | 相机补偿、FB confidence、正确的 H,W,2 光流损失 |
| `runtime.py` | 损失接入、梯度审计、来源标签、诊断 |
| `checkpoint.py` | 仅允许有凭据的 use_flow False→True 与诊断目录迁移 |
| `prepare_data.py` | 训练专用光流/深度缓存；已生成并锁定，勿再编辑 |
| `export.py` | 拼接 PNG、MP4、ZIP、独立 PSNR 与像素核验 |
| `README.md` | 更完整的实现与限制说明 |

原发布光流代码发现 H,W,2 按行归一化/截断、协方差语义以及投影 metadata 梯度像素缩放等问题。新分支修正分量归一化、不裁剪位移，使用稳定 SPD 平方根及经过 CUDA forward 对齐和 gradcheck 的 Torch 投影。**不能称其为逐字复现论文 LPFM**。

## 5. 已完成的证据与提交

实现/协议/预检里程碑已提交：**`c73a97a`**，提交信息 `Add paired training-flow and structural velocity comparison with preflight`。

- `tests/test_flow_comparison_data.py`、`tests/test_flow_comparison_model.py` 与继承的时钟测试合计 **23 passed**（含显式 GPU 检查）。
- 两组各 100 步 GPU 短训，并从 70 步完整恢复到 100 步。
- 20 步 flags-off 短训，保留旧代码路径。
- 光流和动量损失到形变/应力/独立速度头的有限非零梯度检查通过；K2 闲置速度头无梯度。
- 原训练器最终步不执行 optimizer.step，因此另做了真正的优化器更新断言：两组 grid/deformation/stress 均改变，K1 velocity_head 改变、K2 velocity_head 完全不变。检查点未覆盖。

证据：

- `experiments/flow_comparison/preflight.json`
- `experiments/flow_comparison/tests.log`
- `experiments/flow_comparison/optimizer_update_checks.json`
- `output/flow_comparison/verify_optimizer_update.py`
- `experiments/flow_comparison/source_freeze.json`
- `variants/flow_comparison/source_manifest.json`

## 6. 光流数据与冻结约束

训练缓存：`data/derived/flow_comparison_train_only/`，120 张训练图、119 个真实相邻训练帧对；不能复用原数据旧 .flo，因为重划分后的帧对不同。

- UniMatch code commit：`943a99006950ab6c84ea72a72beb147fdd03c19a`。
- 权重：`reproduction/flow-assets/gmflow-scale2-mixdata-train320x576-9ff1c094.pth`。
- 权重 SHA256：`9ff1c094f496f5a209e910b2c84071e712c1f2a1f391f4a4546f1a467328876e`。
- 完整锁：`experiments/flow_comparison/flow_model_lock.json`。
- UniMatch 估计分辨率 800×448，结果缩放到 400×225。
- 相机补偿深度来自仅用训练图像拟合的冻结 H3 重建；**深度和光流均为估计，不是 GT**。
- 只读取 120 张训练 RGB 生成训练监督，没有 val/test RGB 混入；训练帧间隔最大 3 帧，使用真实 dt。
- 每对约 2,010–4,600 个可信像素，中位约 3,272；正反向一致性/不透明前景/深度/边界筛选。
- 目标文件、120 张输入图、H3 教师权重、生成代码等均有 SHA 锁，见缓存 `input_lock.json`、`manifest.json` 与 `experiments/flow_comparison/flow_data_manifest.json`。
- 输入可视检查：`output/flow_comparison/training_flow_inputs.png`。

**不要编辑已经冻结的 `variants/flow_comparison/*.py`、fork、plan 或父级助手文件。**队列在下一阶段/恢复时会检查哈希，修改会使其拒绝继续。发现实质 bug 时先保存状态、归档旧实验和锁文件，写清版本变化再修复；不要静默重写锁文件。尤其不要改 `geometry.py` / `prepare_data.py`，缓存也锁了其代码 SHA。

## 7. 队列会自动完成什么

1. K1 从共同 3k 完整训练到总 50k。
2. K2 从同一个 3k 完整状态独立训练到总 50k。
3. 写 `runs/pidg_flow_comparison/selection_lock.json`，两组按验证 PSNR 选择。
4. 对两组 `best` / `50000` 渲染 50 个最终测试视角并计算四项指标。
5. 每组导出 100 张 GT|Render 拼接 PNG、两段 50 帧视频（10fps 仅展示）、交互 HTML 和 ZIP。
6. 验证拼接左右图像区逐像素一致、独立重算 PSNR、视频帧数/时长、ZIP CRC。
7. 生成 `experiments/flow_comparison/results.json` / `assessment.md`，最终进度变 `completed`。

预计导出路径：

```text
output/flow_comparison/K1_flow_independent/
output/flow_comparison/K2_flow_trajectory/
  GT_Render_best.mp4
  GT_Render_50000.mp4
  pairs/best/00000.png ... 00049.png
  pairs/50000/00000.png ... 00049.png
  index.html
  manifest.json
output/flow_comparison/<模型>_GT_Render.zip
```

每张 PNG 为 3200×950：50px 标题 + 两张 1600×900 图，左 GT、右 Render。原帧 85 对应测试索引 21，即 `pairs/best/00021.png`。

## 8. 训练结束后还必须手工继续的工作

新增辅助脚本与本交接一并保存为独立提交，位于 `output/`，不会改变冻结训练代码。`status.py` 已实际使用；下面三项最终后处理只做过语法检查，**尚未执行最终模型诊断，不能宣称已通过**。

```powershell
# 仅在训练/渲染队列 completed 且 GPU 已释放后顺序运行。
. ./reproduction/scripts/enable_cuda.ps1
envs/pidg-cu128/python.exe -u output/flow_comparison/evaluate_motion.py
envs/pidg-cu128/python.exe -u output/flow_comparison/gradient_balance.py
envs/pidg-cu128/python.exe -u output/flow_comparison/finalize.py
```

### evaluate_motion.py

- 在选择锁定后，用 30 个“验证源帧→下一训练帧”新配对生成原始 UniMatch 光流作诊断，**不反馈训练/选择**。
- 直接比较 source→target 相机下的原始二维光流，**不使用 H3 估计深度补偿**。
- 使用固定观测 confidence mask；模型漏覆盖不能通过删除像素降低误差。
- 比较 geometric、velocity advection、仅相机运动/物体静止 baseline 的 pixel EPE 与 normalized L1。
- 对两组 best/final 共四快照评估；并测量速度/加速度差分步长减半/加倍的敏感性。
- 缓存：`data/derived/flow_comparison_validation_eval`；输出 `experiments/flow_comparison/motion_validation.json`。
- **重要加载修正已写入但待执行核验**：`load_ply()` 不恢复速度头/ID；本脚本显式加载 `material.pt` 中 `velocity_net`、`particle_indices`、`force_sigma`。不可删掉。
- 仍是估计的二维运动观测，不是三维速度 GT。

### gradient_balance.py

- 对验证选中的 best，在固定训练源帧 2/83/155 做 .01 光流/.01 动量加权梯度范数与 cosine 探针。
- 测 deformation、velocity_head、stress_head、physics_shared、rate_coefficient；不执行优化器更新。
- 追加日志重定向到 `output/flow_comparison/gradient_probes/`，避免混入真实训练日志。
- 输出 `experiments/flow_comparison/gradient_balance.json`。
- 只是局部固定探针，不是全部随机训练批次的审计；不能单凭范数认定唯一因果。

### finalize.py

- 核验源码/数据/共同起点哈希、两组仅 velocity_source 不同、最终 50k 完整检查点、闲置时钟/速度头未改变、参数有限、应力头已更新、训练光流配对合法、导出核验。
- 预期每组动量损失计算 420 次，其中原训练器最终步跳过优化器更新，实际参与更新为 419 次；还断言两组训练光流帧调度相同。若断言失败，先查日志与 RNG/调用路径，不能为了通过而删断言。
- 画 `training_comparison.png`、`motion_comparison.png`；输出 `experiments/flow_comparison/final_verification.json`。
- 可能还需根据实际日志结构修正辅助脚本。例如历史诊断文件是否存在；必须据证据处理，不能伪造通过。

### 最后交付

1. 看实际 best/final GT|Render PNG，至少检查早/中/晚时间与原帧 85；查看图表和视频。
2. 在 `assessment.md` 加入运动诊断、差分敏感性和梯度探针的真实结果与限制。
3. 解释图像 PSNR、观测二维运动、内部速度一致性三者是否同时改善。
4. 如结果差，列出有证据的原因与后续可区分原因的实验；不要直接断言一定是缺外力，也不要立刻加所有未知量联合训练。
5. 单独提交结果里程碑，显式 stage 本任务报告、辅助脚本/图表/验证收据；不要提交巨量 checkpoint/视频/PNG，也不要 stage 无关脏文件。
6. 中文给用户简短结果表、报告链接、两组 best 拼接视频/浏览页链接；明确还有哪些限制。

用户追加授权：根据比较继续一个受控改进。若综合效果良好，沿当前技术路径处理一个有证据的瓶颈；若不行，尝试形变和应力共享 Grid4D/latent、速度由完整轨迹导出的同一分支。事先记录的门槛、结构含义和测试要求见 `experiments/flow_comparison/followup_plan.md`。不要把本次两组交付当作取消后续授权；下一组必须独立变体、测试、短训、完整训练和 GT|Render 比较，保留 K1/K2。

## 9. 历史结果与研究背景（勿重跑）

- 历史 H3：`runs/pidg_integrity_tuning/H3_combined/rubber_duck`。
  最终 **测试** PSNR 40.878231、LPIPS .009441059；不是当前 K1 的验证 PSNR，不能直接拿混合 split 数字比较。
- J1：`runs/pidg_monotonic_training/J1_monotonic_kinematic/rubber_duck`。
  单调时钟 + .001 软一致性，已完整 50k；验证选中 49k，测试 best PSNR 40.080849，final 40.071724；最终内部相对速度误差约 99.73%。
  报告 `experiments/monotonic_training/assessment.md`，GT/Render 在 `output/monotonic_training/J1_monotonic_kinematic/`。
- 冻结轨迹拟合实验已完成，见 `experiments/frozen_velocity_probe` 与 `variants/frozen_velocity_probe`。
  原物理网络内部可学习标量时钟会折叠；固定单调时钟后全目标训练相对误差约 20.26%、held-time 21.97%，小任务可到 2.16%。不能说表示完全不能拟合。
- 冻结拟合约 49,152,000 采样对；J1 联训约 1,202,944，监督预算差约 40.86 倍，且目标移动、Huber/CMR 等不同；不能把拟合失败唯一归于结构。
- 最终测试在历史实验中已有调参暴露。本轮按验证选择仍不能称它为“全新未暴露测试集”。
- 原论文 LPFM 与本地 Rubber Duck 发布配置不同：本地该场景 `use_flow=False`；当前补光流是新的实验，不冒称之前已经使用完整论文监督。

## 10. 环境与工作区安全

- Python：`envs/pidg-cu128/python.exe`；Torch 2.8 + cu128。
- GPU/C++ 环境：先 dot-source `reproduction/scripts/enable_cuda.ps1`。
- hash source/build 由启动器指向 `third_party/PIDG/hashencoder` / `third_party/PIDG/tmp_build`，已有编译产物；不要随意重建环境。
- 启动器：`scripts/pidg_variant_entrypoint.py`；训练全状态：`reproduction/scripts/training_checkpoint.py`。
- 基线代码概念映射：`PIDG_REPO_MAP.md`；根 `AGENTS.md` 的模块开关、保留 baseline、测试与分里程碑提交规则继续适用。
- Git 当前 `main`，本任务实现提交 `c73a97a`；本交接和后处理准备为之后的独立提交，见最新 `git log`。仓库原有大量无关 untracked 文件和三个原有 tracked 改动：`experiments/integrity_feedback/proposals.json`、`experiments/results.json`、`experiments/results.md`。**不要覆盖、reset、clean 或一并提交。**
- 当前未提交的本任务运行文件主要为进度、监控页和协调器/数据生成日志。不是丢失的工作，不要删除。
- 沙箱 `.git` 写入可能需 `require_escalated` 自动审核；此前 scoped git add/commit 均通过。无需向用户重复索取已授权的常规提交确认。
- Windows 文件移动/删除使用单一 PowerShell 流程与 `-LiteralPath`，先验证目标在工作区；禁止宽泛递归删除。
- 网络资源已获取，不需要再次下载模型，更不能上传数据。不要输出 SSH 私钥/密码。

## 11. 可直接粘贴给下一位 Codex 的提示

> 请读取 `F:\A_Edinburgh\PhysGauss\CODEX_HANDOFF_FLOW_COMPARISON_20261102.md` 和根 `AGENTS.md`，先检查现有 K1/K2 光流对照队列进度，勿重复启动。接续已经授权的两组完整训练、选择锁定、GT/Render 导出，然后顺序执行并核验尚未运行的运动诊断/梯度平衡/最终验证脚本，解释两组结果并提交独立结果里程碑。保留原 PIDG、历史模型、冻结协议和无关改动。若队列仍活跃，继续监控，不要同时启动 GPU 后处理。
