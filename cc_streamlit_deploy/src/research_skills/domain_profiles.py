"""Variable-driven scientific profiles shared as evidence context by all Skills.

The profiles describe domain vocabulary, mechanisms, candidate model families,
and measurements. They do not contain question-specific answers or citations;
each Skill still performs its own reasoning over the current EvidenceBundle.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.research_skills.contracts import SkillInput


@dataclass(frozen=True)
class DomainProfile:
    key: str
    title: str
    independent: tuple[str, ...]
    dependent: tuple[str, ...]
    direct_claim: str
    mechanism_chain: str
    boundary: str
    gap_focus: str
    model: str
    model_note: str
    variables_and_units: str
    measurements: str
    factor_design: str
    predictions: str
    falsifiers: tuple[str, str]
    confounders: str


PROFILES: dict[str, DomainProfile] = {
    "formula_models": DomainProfile(
        "formula_models", "疲劳模型预测对象比较",
        ("Basquin应力—寿命输入", "Murakami缺陷/硬度输入", "Paris或Walker裂纹驱动力"),
        ("S-N寿命Nf", "疲劳极限σw", "裂纹扩展速率da/dN"),
        "Paris关系描述稳定裂纹扩展区的da/dN而不能直接预测总疲劳寿命；Basquin面向S-N寿命，Murakami类模型面向缺陷控制疲劳极限。预测对象和输入未统一时不能给出哪一个更准确的数值排名。",
        "Paris/Walker把裂纹驱动力映射为扩展速率，寿命预测还需对裂纹长度积分且另行处理起裂；Basquin拟合应力—寿命，Murakami把硬度与√area映射为疲劳极限。",
        "必须统一预测对象、材料状态、单位、R、环境、裂纹阶段、拟合区间和验证数据，才能比较误差。",
        "缺少同一数据集上按各自正确预测对象校准、再映射到共同终点的外部验证。",
        "da/dN=C(ΔK)^m；Basquin: σa=σf′(2Nf)^b；Murakami类: σw=f(HV,√area,R,location)",
        "公式结构用于区分预测对象；任何系数、指数和修正项都必须来自对应文献或实验拟合。",
        "da/dN：m/cycle；ΔK：MPa√m；σa、σw：MPa；Nf：cycle；√area：µm；HV：硬度标尺值。",
        "统一数据字典、裂纹长度测量、S-N试验、硬度/缺陷表征和独立验证集。",
        "先按各模型原生终点校准，再通过明确的寿命积分或极限判据映射到共同终点；禁止直接混排原始输出。",
        "正确使用时各模型应在自身预测对象上通过外部验证；跨对象比较必须报告映射误差。",
        ("模型在其原生预测对象上不能优于无机制基线。", "所谓精度排名在统一对象、单位和验证集后消失或反转。"),
        "材料批次、R、表面、环境、裂纹阶段、单位和拟合区间。",
    ),
    "defect_size_life": DomainProfile(
        "defect_size_life", "缺陷尺寸与疲劳寿命",
        ("孔隙尺寸√area", "缺陷位置d/√area", "应力幅σa"), ("疲劳寿命Nf",),
        "在可比的L-PBF Ti-6Al-4V条件内，较大的√area通常对应更短的疲劳寿命Nf，但尺寸效应受缺陷位置、表面状态、残余应力和HIP状态调节，不能解释为单一阈值。",
        "√area增大使等效初始裂纹尺度和局部驱动力提高，起裂循环数减少；自由表面、拉伸残余应力和粗糙度可放大该效应，HIP闭合缺陷或压缩残余应力可削弱它。",
        "必须匹配材料批次、HIP/热处理、表面状态、σa、R、HCF/VHCF区间及run-out定义。",
        "同批次下√area与d/√area的独立贡献、交互项以及HCF/VHCF边界尚需配对验证。",
        "log10(Nf)=β0−β1log10(σa)−β2log10(√area)+β3(d/√area)+β4R+β5σres+β6[log10(√area)×d/√area]+ε",
        "待拟合寿命交互模型，并非文献已确定公式；含删失时应与生存模型比较。",
        "Nf：cycle；σa、σres：MPa；√area、d：µm；R与d/√area：无量纲。",
        "疲劳前XCT、断口SEM、三维表面形貌、XRD残余应力和寿命/run-out记录。",
        "按√area连续分布或分层，并与表面/近表面/内部位置交叉；所有组匹配表面、HIP和载荷。",
        "尺寸主效应、尺寸×位置交互及外部批次预测均应按预注册方向改善。",
        ("控制载荷、位置和表面后，√area项不改善留出批次预测。", "尺寸×位置交互方向在独立批次中反转或无法复现。"),
        "缺陷形貌、表面粗糙度、残余应力、组织尺度、HIP和应力比。",
    ),
    "defect_location": DomainProfile(
        "defect_location", "缺陷尺寸—位置耦合",
        ("孔隙尺寸√area", "缺陷距表面距离d", "归一化深度d/√area"), ("疲劳寿命Nf", "裂纹起源概率"),
        "孔隙位置会导致相同√area对应不同疲劳寿命：表面和近表面缺陷更易与自由表面应力场及粗糙度缺口耦合，内部缺陷则受包围材料、内部起裂与环境隔离条件控制。",
        "自由表面降低裂纹形成约束并改变局部应力强度；d/√area下降时缺陷场与自由表面相互作用增强，主裂纹源概率可上升。",
        "比较必须固定缺陷形貌、表面加工、σa、R、HIP、残余应力与疲劳区间，且统一表面/近表面分类规则。",
        "缺少同尺寸、不同深度的配对缺陷以及尺寸×位置交互的独立批次检验。",
        "log10(Nf)=β0+β1log10(√area)+β2(d/√area)+β3log10(√area)·(d/√area)+u_batch+ε",
        "待拟合位置交互模型；裂纹源类别另用多项Logistic或竞争风险模型。",
        "√area、d：µm；d/√area：无量纲；Nf：cycle；裂纹源：分类变量。",
        "疲劳前XCT定位、表面轮廓、断口SEM反查主裂纹源、寿命删失记录。",
        "保持√area匹配，设置表面、近表面、内部位置层，并加入独立制造批次验证。",
        "位置主效应及尺寸×位置交互应同时改变Nf和主裂纹源概率。",
        ("同√area配对后，位置项对Nf和起源概率均无增量解释力。", "XCT位置与断口确认的主裂纹源不一致且无系统方向。"),
        "缺陷形貌、表面Ra/Rz、局部残余应力、载荷梯度和环境。",
    ),
    "surface_competition": DomainProfile(
        "surface_competition", "表面粗糙度与近表面缺陷竞争起裂",
        ("Ra/Rz及三维形貌", "近表面缺陷√area与深度", "表面加工状态"), ("裂纹起裂位置", "起裂寿命Ni"),
        "机械加工后，表面粗糙度与近表面缺陷竞争控制疲劳裂纹起裂：主导源取决于加工沟槽局部缺口效应与近表面缺陷驱动力谁更大，不能默认孔隙始终主导。",
        "粗糙峰谷提高表面局部应力并促进表面起裂；加工去除表层缺陷后，较深缺陷可能不再临界，而残留近表面缺陷可在表面改善后成为新的主导源。",
        "必须同时报告去除量、Ra/Rz或面积参数、缺陷深度、加工残余应力、σa、R和起裂判据。",
        "缺少同一试样族中表面形貌与近表面缺陷的联合定量及主导区转换边界。",
        "P(defect-origin)=logit⁻¹[β0+β1Krough+β2Kdefect+β3(Krough·Kdefect)+u_batch]",
        "待拟合起裂源概率模型；Krough与Kdefect必须由实测形貌/缺陷定义，不能预填阈值。",
        "Ra、Rz、Sa、Sz、√area、深度：µm；起裂源概率：无量纲；Ni：cycle。",
        "三维白光形貌、XCT、XRD、复制法或原位成像、断口SEM盲法定位。",
        "机械加工量×近表面缺陷层析分组，并设置表面形貌相近但缺陷不同及缺陷相近但粗糙度不同的匹配对照。",
        "主导源应随Krough/Kdefect相对量级发生可重复转换。",
        ("控制缺陷后表面形貌不影响起裂源，且控制形貌后缺陷也无效。", "预测的表面/缺陷主导转换在独立批次中不能复现。"),
        "加工硬化、残余应力、去除深度、缺陷形貌、载荷梯度。",
    ),
    "hip_dependence": DomainProfile(
        "hip_dependence", "HIP疲劳增益的条件依赖性",
        ("HIP状态", "表面状态", "载荷/疲劳区间"), ("疲劳寿命Nf", "疲劳极限或起裂位置"),
        "HIP并非在所有条件下都必然提高疲劳性能：其缺陷闭合作用在内部缺陷主导时更可能显现，而as-built表面缺口、组织粗化或不利表面状态可限制收益。",
        "HIP闭合孔隙并松弛部分残余应力，但同时改变α/β组织尺度；若自由表面缺口仍控制起裂，内部缺陷减少未必转化为寿命增益。",
        "必须分层比较as-built与machined/polished表面、HIP制度、组织、残余应力、R、σa及HCF/VHCF。",
        "HIP×表面状态×载荷区间的交互效应，以及缺陷闭合收益与组织粗化代价的竞争边界仍需匹配试验。",
        "g(Nf)=β0+β1HIP+β2Surface+β3Regime+β4(HIP·Surface)+β5(HIP·Regime)+u_batch",
        "待拟合交互模型；Nf有run-out时使用生存链接函数，不预设固定HIP收益。",
        "HIP/表面/区间：分类变量；Nf：cycle；疲劳极限：MPa；组织尺度：µm。",
        "XCT缺陷闭合、EBSD/金相组织、XRD残余应力、表面形貌和分层疲劳试验。",
        "HIP与非HIP分别交叉as-built/machined表面，并按载荷区间分层；同批材料验证。",
        "HIP增益应随内部缺陷控制比例上升，而表面控制组收益可减弱或消失。",
        ("HIP组缺陷减少但匹配表面条件下Nf无改善，且机制指标不支持预期链条。", "HIP效应方向完全由表面或批次解释，交互项不能外部复现。"),
        "HIP温压时间、冷却路径、表面缺口、组织粗化、残余应力和批次。",
    ),
    "residual_short_crack": DomainProfile(
        "residual_short_crack", "残余应力与短裂纹有效驱动力",
        ("残余应力σres", "外加ΔK", "裂纹长度a"), ("短裂纹da/dN", "ΔKeff"),
        "残余拉应力通常提高局部平均应力、削弱裂纹闭合并增大ΔKeff，从而可能加快短裂纹扩展；但短裂纹闭合尚未充分建立且残余应力会循环松弛，不能直接套用长裂纹Paris参数。",
        "σres改变裂纹尖端开闭载荷与有效驱动力；裂纹增长和循环塑性又会重新分布σres，因此影响随a和循环数演化。微观组织屏障可使相同ΔKeff下的短裂纹速率离散。",
        "必须匹配初始σres场、裂纹长度、R、ΔK、载荷历史、组织和表面，并原位或分阶段复测应力松弛。",
        "短裂纹阶段ΔKeff的可测定义、σres松弛与组织屏障的相对贡献尚缺同试样同步测量。",
        "log(da/dN)=β0+β1log(ΔKeff)+β2σres+β3log(a)+β4R+u_specimen",
        "待拟合短裂纹混合效应模型；Paris关系仅作长裂纹基线，不能直接外推。",
        "da/dN：m/cycle；ΔKeff：MPa√m；σres：MPa；a：m或mm但全程统一；R：无量纲。",
        "XRD/孔钻或适用应力测量、复制法/DIC/原位成像、裂纹开闭载荷和EBSD。",
        "拉伸、近零及压缩残余应力层，与多个短裂纹长度阶段交叉；保持外加R和ΔK历史一致。",
        "σres方向应系统改变开闭载荷、ΔKeff与da/dN，并随循环松弛而衰减。",
        ("ΔKeff匹配后σres不再解释da/dN，且开闭载荷不随σres变化。", "独立试样中效应仅随组织取向出现，残余应力方向不能复现。"),
        "裂纹闭合、循环松弛、α片层/织构、初始裂纹长度和载荷顺序。",
    ),
    "microstructure_growth": DomainProfile(
        "microstructure_growth", "多尺度组织对裂纹扩展的共同作用",
        ("α片层宽度lα", "先前β晶粒dβ", "织构强度T"), ("da/dN", "裂纹路径曲折度"),
        "α片层宽度、先前β晶粒和织构通过滑移传递、裂纹偏转及晶界/相界屏障共同影响da/dN；各尺度可能竞争，不能用孔隙尺寸公式代替组织模型。",
        "lα与dβ改变屏障间距和裂纹路径；织构控制有利滑移系取向。屏障增强可降低局部扩展速率，而连续易滑移取向或基面滑移可形成加速通道。",
        "必须匹配ΔK或ΔKeff、R、裂纹阶段、热处理、残余应力及取样方向，并区分短裂纹与长裂纹。",
        "缺少在相同载荷历史下同时量化lα、dβ、织构与裂纹路径的交互模型和跨批次验证。",
        "log(da/dN)=β0+β1log(ΔKeff)+β2lα+β3dβ+β4T+β5(lα·T)+u_batch",
        "待拟合组织交互模型；若证据不足以选函数形式，应比较线性、幂律和分段候选。",
        "da/dN：m/cycle；ΔKeff：MPa√m；lα、dβ：µm；T：预注册无量纲织构指标。",
        "EBSD取向/晶粒、金相片层定量、裂纹复制或原位成像、路径曲折度与断口分析。",
        "以热处理获得组织梯度，按lα×织构层分组并测量dβ协变量；加载方向相对织构固定。",
        "组织项应在控制ΔKeff后解释da/dN及路径曲折度，并表现出可重复交互。",
        ("控制ΔKeff和取向后，所有组织项均无增量解释力。", "组织预测方向在独立取向或批次中系统反转且无机制指标响应。"),
        "热处理、残余应力、裂纹阶段、加载方向、表面状态和初始裂纹长度。",
    ),
    "orientation": DomainProfile(
        "orientation", "建造方向的独立疲劳效应",
        ("建造方向", "缺陷取向", "织构"), ("疲劳寿命/极限",),
        "控制表面和热处理后，建造方向仍可能通过缺陷取向、织构与残余应力影响疲劳性能；若这些中介量也被匹配，表观方向差异可能显著减弱。",
        "建造方向改变熔池/层间缺陷相对主应力的取向、晶体织构和残余应力张量；三者共同改变起裂概率和早期裂纹路径。",
        "必须同批制造并匹配表面、热处理、缺陷统计、加载方向、试样几何、R和疲劳区间。",
        "在缺陷取向、织构和残余应力均实测并纳入后，建造方向是否仍有直接效应尚需中介分析。",
        "g(Nf)=β0+β1Orientation+β2DefectOrientation+β3Texture+β4σres+u_batch",
        "待拟合中介/分层模型；直接效应与经缺陷、织构、应力传递的间接效应必须分开。",
        "Nf：cycle；疲劳极限：MPa；σres：MPa；取向角：degree；织构：无量纲。",
        "XCT缺陷取向、EBSD织构、XRD应力、表面形貌、S-N或生存疲劳试验。",
        "至少两个建造方向，同批次、同表面和同热处理；逐步匹配中介量并用独立批次验证。",
        "若存在独立效应，加入全部中介量后Orientation项仍改善预测。",
        ("中介量纳入后Orientation项消失且留出预测不改善。", "方向效应随批次或加载轴改变而反转，无法归因于稳定机制。"),
        "缺陷取向、织构、残余应力、表面、几何和载荷轴。",
    ),
    "stress_ratio_growth": DomainProfile(
        "stress_ratio_growth", "应力比对裂纹扩展与阈值的影响",
        ("应力比R", "ΔK/ΔKeff"), ("da/dN", "ΔKth", "Paris C与m"),
        "应力比R升高通常通过减弱裂纹闭合提高同一外加ΔK下的ΔKeff，并可改变表观da/dN、ΔKth及拟合的Paris C/m；不同R下参数只有在裂纹阶段、环境和拟合区间一致时才能比较。",
        "R改变Kmin/Kmax与开闭载荷，因而改变有效驱动力；用Walker或ΔKeff归一化可减少但未必消除R依赖。",
        "必须统一长/短裂纹、ΔK控制方式、频率、环境、温度、闭合测量和Paris拟合区间。",
        "不同R修正形式对阈值区、Paris区和短裂纹区的适用边界及参数可迁移性仍需同数据比较。",
        "da/dN=C(ΔK)^m；或 da/dN=C[ΔK(1−R)^γ]^m",
        "前者为Paris型长裂纹关系，后者为Walker型候选修正；C、m、γ需在统一单位和区间拟合。",
        "da/dN：m/cycle；ΔK、ΔKth：MPa√m；R、m、γ：无量纲；C单位随m变化。",
        "裂纹长度/柔度法、闭合载荷测量、阈值降载程序、统一Paris区回归。",
        "多个R水平交叉阈值区与Paris区；短裂纹另设组，禁止与长裂纹参数混合。",
        "R效应应在外加ΔK下明显，在ΔKeff或Walker修正后收敛程度提高。",
        ("统一ΔKeff后R仍呈无规律强偏差且修正模型无改进。", "C/m差异仅由不同拟合区间或单位造成，统一后消失。"),
        "裂纹闭合、环境、频率、短/长裂纹、载荷历史和拟合区间。",
    ),
    "regime_transition": DomainProfile(
        "regime_transition", "HCF到VHCF的裂纹起源转换",
        ("寿命区间HCF/VHCF", "缺陷位置", "环境隔离"), ("表面/内部起裂概率", "Nf"),
        "从HCF进入VHCF会影响裂纹起源位置并可能使其从表面转向内部：表面起裂并非必然消失，但内部缺陷或组织不连续处拥有更长时间成为主裂纹源，内部环境隔离和fish-eye型扩展区也会改变早期扩展过程。",
        "低应力幅延长起裂竞争时间；表面小缺口若未达临界，内部缺陷可在超长循环内累积损伤并形成fish-eye类区域。",
        "HCF与VHCF必须按应力幅、频率、温升、表面状态、缺陷位置和run-out/失效定义分别分析。",
        "表面与内部起源概率随寿命区间连续转换的边界，以及频率/环境对fish-eye形成的调节仍需统一试验。",
        "P(internal-origin)=logit⁻¹[β0+β1log10(Nf)+β2σa+β3Depth+β4Environment+u_batch]",
        "待拟合起裂源概率模型；不能把HCF和VHCF寿命数据不分层合并。",
        "Nf：cycle；σa：MPa；缺陷深度：µm；起裂源与环境：分类变量。",
        "超声/常规疲劳温升监测、XCT、断口SEM、fish-eye尺寸和起源盲法分类。",
        "跨HCF/VHCF的应力层级，保持表面与材料批次一致，分层记录表面/内部起裂。",
        "随Nf区间延伸，内部起裂概率上升并伴随可核验的内部断口特征。",
        ("控制应力幅和缺陷分布后，寿命区间不改变起裂源概率。", "所谓转换完全由试验频率温升或表面批次差异解释。"),
        "频率温升、表面缺口、内部缺陷、环境、run-out处理和设备差异。",
    ),
    "environment": DomainProfile(
        "environment", "环境介质对起裂与扩展的阶段性影响",
        ("空气/真空/腐蚀介质", "频率f", "温度T"), ("起裂寿命Ni", "da/dN", "总寿命Nf"),
        "空气、真空和腐蚀环境对起裂与扩展的作用不同：氧化、吸附或氢相关过程可改变表面反应与裂纹尖端损伤，且效应受频率和温度控制；本地证据不足时不能给出统一数值排序。",
        "环境影响表面膜形成、裂纹尖端化学反应和氢扩散；较低频率提供更长反应时间，温度同时改变反应速率和材料变形机制。",
        "必须分别报告介质成分/压力、频率、温度、R、裂纹阶段、表面状态和暴露时间。",
        "同材料状态下环境×频率×温度对起裂与扩展阶段贡献的分离，以及真空/腐蚀边界仍缺匹配数据。",
        "g(Y)=β0+β1Environment+β2log(f)+β3T+β4Environment·log(f)+β5Environment·T+u_batch",
        "待拟合环境交互模型；Y应分别取Ni、da/dN或Nf，不能把不同预测对象混为一个终点。",
        "Ni、Nf：cycle；da/dN：m/cycle；f：Hz；T：°C或K；介质/压力按实验原单位。",
        "受控气氛/真空/腐蚀腔、温度和频率校准、复制法或原位裂纹测量、表面化学与断口分析。",
        "环境介质×频率×温度因子设计；起裂和扩展终点分开记录，材料与载荷批次内随机化。",
        "环境效应应随反应时间尺度改变，并在Ni与da/dN上呈可区分的阶段响应。",
        ("改变介质后Ni和da/dN均无可重复差异，且表面/氢指标不响应。", "环境差异在统一频率和温度后消失，原效应由热或速率混杂解释。"),
        "介质纯度/压力、频率、温度、表面膜、氢含量、R和裂纹阶段。",
    ),
}


def _variables(value: SkillInput) -> set[str]:
    frame = value.query_frame or (value.evidence_bundle or {}).get("query_frame") or {}
    return set(frame.get("independent_variables") or []) | set(frame.get("dependent_variables") or [])


def select_domain_profile(value: SkillInput) -> DomainProfile:
    """Select a profile from parsed scientific entities, never exact questions."""
    variables = _variables(value)
    frame = value.query_frame or (value.evidence_bundle or {}).get("query_frame") or {}
    requested_formulas = set(frame.get("requested_formulas") or [])
    if "environmental_medium" in variables or frame.get("environment"):
        return PROFILES["environment"]
    if "fatigue_regime" in variables and "crack_origin_location" in variables:
        return PROFILES["regime_transition"]
    if "stress_ratio" in variables and variables & {"delta_k_threshold", "paris_parameters", "crack_growth_rate_da_dn"}:
        return PROFILES["stress_ratio_growth"]
    if requested_formulas:
        return PROFILES["formula_models"]
    if "build_orientation" in variables:
        return PROFILES["orientation"]
    if variables & {"alpha_lath_width", "prior_beta_grain", "crystallographic_texture"}:
        return PROFILES["microstructure_growth"]
    if "residual_stress" in variables and variables & {"short_crack_growth_rate", "effective_delta_k", "crack_growth_rate_da_dn"}:
        return PROFILES["residual_short_crack"]
    if "hip" in variables:
        return PROFILES["hip_dependence"]
    if "surface_roughness" in variables and "near_surface_defect" in variables:
        return PROFILES["surface_competition"]
    if "pore_location" in variables:
        return PROFILES["defect_location"]
    return PROFILES["defect_size_life"] if "pore_size" in variables else DomainProfile(
        "generic", "问题变量的条件化疲劳关系",
        tuple(frame.get("independent_variables") or ("题目自变量",)),
        tuple(frame.get("dependent_variables") or ("题目因变量",)),
        "现有正式证据只支持在已报告条件内进行条件化判断，不能跨材料状态、裂纹阶段或载荷条件直接外推。",
        "需要把观测关系拆分为局部驱动力、组织屏障和载荷历史三个可测环节，并区分相关性与因果链。",
        "材料、制造、热处理、表面、R、频率、温度、环境和疲劳阶段必须逐项核对。",
        "尚缺同批次匹配条件下对主效应、竞争机制和适用边界的联合检验。",
        "g(y)=β0+Σβi·xi+Σβij·xi·xj+u_batch+ε",
        "仅为待拟合候选模型；若证据不足，应明确拒绝确定函数形式。",
        "各变量保持原始SI单位并报告测量不确定度。",
        "按问题变量选择校准表征方法、疲劳终点和机制中介量。",
        "基线、主效应、关键交互和独立批次验证四层设计。",
        "主效应与机制指标应同步变化，并改善独立批次预测。",
        ("主效应及交互项均不改善外部预测。", "预测方向在独立批次中稳定反转或机制指标不响应。"),
        "材料批次、表面、热处理、残余应力、组织和载荷历史。",
    )


def profile_prompt_block(value: SkillInput) -> str:
    profile = select_domain_profile(value)
    return (
        f"领域画像：{profile.title}\n"
        f"必须显式覆盖的自变量：{'、'.join(profile.independent)}\n"
        f"必须显式覆盖的因变量：{'、'.join(profile.dependent)}\n"
        f"候选机制：{profile.mechanism_chain}\n"
        f"模型族：{profile.model}（{profile.model_note}）\n"
        f"边界：{profile.boundary}\n"
        "该画像只提供术语、机制候选和实验结构；所有事实方向仍须由EvidenceBundle核对。"
    )
