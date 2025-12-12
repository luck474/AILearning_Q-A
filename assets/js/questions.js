const courses = [
  {
    "id": "deep_learning",
    "name": "深度学习 (Deep Learning)",
    "questions": [
      {
        "id": 1,
        "question": "上学期学习的利用线性回归预测加州房价的例子，该线性回归模型可以看做是一个单层的神经网络。",
        "type": "true_false",
        "options": [
          "True",
          "False"
        ],
        "answer": "True"
      },
      {
        "id": 2,
        "question": "线性神经网络的训练过程中，如果我们将权重初始化为零，会发生什么。算法仍然有效吗？",
        "type": "multiple_choice",
        "options": [
          "A.权重初始化为零是理想选择，算法能快速收敛",
          "B.模型无法打破对称性，导致神经元行为一致，算法性能严重下降",
          "C.梯度爆炸问题会出现，但算法仍可正常训练",
          "D.权重初始化为零会导致所有输出为零，梯度无法更新，算法完全失效"
        ],
        "answer": "B"
      },
      {
        "id": 3,
        "question": "以下关于Softmax回归和Logistic回归区别的描述中，哪一项是正确的？",
        "type": "multiple_choice",
        "options": [
          "A.Softmax回归是Logistic回归的一种特例，专门用于处理多类别分类问题。",
          "B.Logistic回归使用Sigmoid函数作为激活函数，而Softmax回归使用ReLU函数作为激活函数。",
          "C.Logistic回归是二分类模型，其输出是一个标量概率值；Softmax回归是多分类模型，其输出是一个概率分布向量。",
          "D.两者均使用均方误差损失函数（Mean Squared Error）来训练模型。"
        ],
        "answer": "C"
      },
      {
        "id": 4,
        "question": "Pytorch中，训练模型的过程中用于计算梯度的函数是？",
        "type": "multiple_choice",
        "options": [
          "A.backward( )",
          "B.gred_( )",
          "C.update( )",
          "D.forward( )"
        ],
        "answer": "A"
      },
      {
        "id": 5,
        "question": "两个多层感知机模型：模型A（1个隐藏层，10个神经元）和模型B（3个隐藏层，每层20个神经元）。模型B的层数更深、神经元更多，因此其容量和表示能力一定强于模型A。",
        "type": "true_false",
        "options": [
          "True",
          "False"
        ],
        "answer": "False"
      },
      {
        "id": 6,
        "question": "在训练MLP时使用了Dropout技术。在训练时随机“丢弃”了一部分神经元。在模型预测阶段，为了模拟训练时大量神经元的随机组合效应，也应该继续随机丢弃一部分神经元。",
        "type": "true_false",
        "options": [
          "True",
          "False"
        ],
        "answer": "False"
      },
      {
        "id": 7,
        "question": "一个多层感知机，在训练集上准确率达到99%，但在测试集上只有70%。这可能是因为模型过拟合了，以下哪种方法不能有效防止过拟合？",
        "type": "multiple_choice",
        "options": [
          "A.增加训练数据量",
          "B.使用更复杂的模型（如增加隐藏层神经元数量）",
          "C.应用权重衰减（L2正则化）",
          "D.使用dropout技术"
        ],
        "answer": "A"
      },
      {
        "id": 8,
        "question": "使用Sigmoid函数作为一个10层MLP的激活函数，但发现训练初期梯度非常小，网络学习缓慢。这种现象最可能的原因是什么？",
        "type": "multiple_choice",
        "options": [
          "A.学习率设置过大",
          "B.发生了梯度爆炸",
          "C.模型陷入了局部最优",
          "D.发生了梯度消失"
        ],
        "answer": "D"
      },
      {
        "id": 9,
        "question": "关于“隐藏层”，以下哪个描述是最准确的",
        "type": "multiple_choice",
        "options": [
          "A.它的神经元数量是保密的，所以叫“隐藏”层",
          "B.它的输出值在训练数据中没有直接给出标签，是模型内部的特征表示",
          "C.它的权重在训练过程中是不可见的",
          "D.激活函数是ReLU"
        ],
        "answer": "B"
      },
      {
        "id": 10,
        "question": "在模型中使用了一个前向线性层，该层继承nn.Module，在init方法中通过nn.Parameter(torch.randn(3,3))初始化参数，并在forward方法中用该参数实现输入张量与权重的矩阵乘法运算。这个参数不会参与梯度计算",
        "type": "true_false",
        "options": [
          "True",
          "False"
        ],
        "answer": "False"
      },
      {
        "id": 11,
        "question": "在训练房价预测模型时，构建了 “线性层（8 维→ 4 维）→ReLU→线性层（ 4 维→ 1 维）” 的顺序组合网络。可以用nn.Sequential实现",
        "type": "true_false",
        "options": [
          "True",
          "False"
        ],
        "answer": "True"
      },
      {
        "id": 12,
        "question": "一个复杂模型在CPU上训练太慢，想转移到GPU上加速。为保证正常运行，需要确保以下哪项？",
        "type": "multiple_choice",
        "options": [
          "A.模型参数移动到GPU",
          "B.输入数据移动到GPU",
          "C.损失函数计算在GPU上进行",
          "D.以上所有"
        ],
        "answer": "D"
      },
      {
        "id": 13,
        "question": "在PyTorch中构建一个10层神经网络，要求第2层和第8层共享相同的参数（即在训练过程中始终保持一致）。以下哪种实现方式最正确、可靠、高效？",
        "type": "multiple_choice",
        "options": [
          "A.",
          "class Net(nn.Module):",
          "def __init__(self):",
          "super().__init__()",
          "self.layers = nn.ModuleList([nn.Linear(100, 100) for _ in range(10)])",
          "# 在初始化后手动复制参数",
          "self.layers[7].weight.data.copy_(self.layers[1].weight.data)",
          "self.layers[7].bias.data.copy_(self.layers[1].bias.data)"
        ],
        "answer": "A"
      },
      {
        "id": 14,
        "question": "B.",
        "type": "unknown",
        "options": [],
        "answer": ""
      },
      {
        "id": 15,
        "question": "C.",
        "type": "unknown",
        "options": [],
        "answer": ""
      },
      {
        "id": 16,
        "question": "D.",
        "type": "unknown",
        "options": [],
        "answer": ""
      },
      {
        "id": 17,
        "question": "在深度学习的早期研究以及现在的一些代码中，我们常常会看到使用标准差为0.01的高斯随机分布（即 weight = torch.randn(fan_in, fan_out) * 0.01）来初始化神经网络的权重。关于这种做法的原因和影响，以下哪个描述是最准确的？",
        "type": "multiple_choice",
        "options": [
          "A.选择0.01这个数值是为了确保所有参数在初始化时都是正值，从而加速模型收敛。",
          "B.这是一种简单的启发式方法，目的是让权重从一个接近零但又不完全为零的小范围开始，旨在避免梯度完全消失，同时防止激活值在初始阶段就出现饱和。",
          "C.标准差设为0.01是经过严格数学推导得出的最优值，适用于所有网络结构和激活函数。",
          "D.其主要目的是为了在训练开始时让损失函数达到一个较低的数值，从而减少模型达到收敛所需的总训练时间。"
        ],
        "answer": "B"
      },
      {
        "id": 18,
        "question": "小明正在设计一个识别猫狗图片的分类系统。他选择使用卷积神经网络（CNN）而不是传统的全连接神经网络，主要是希望CNN帮助解决以下哪个实际问题？",
        "type": "multiple_choice",
        "options": [
          "A.减少模型需要训练的权重数量，降低计算成本",
          "B.直接提取图片的颜色分布特征",
          "C.避免使用激活函数",
          "D.将图片压缩成更小的尺寸以便存储"
        ],
        "answer": "A"
      },
      {
        "id": 19,
        "question": "工程师小张正在设计一个卷积神经网络来识别监控视频中的行人。输入视频帧的分辨率为64×64像素，他使用一层卷积层，并希望输出特征图的尺寸保持为64×64不变。已知卷积核尺寸为3×3，步幅（stride）设为1。那么，为了实现输出尺寸不变，他应该将填充（padding）设置为多少？",
        "type": "multiple_choice",
        "options": [
          "A.0",
          "B.1",
          "C.2",
          "D.3"
        ],
        "answer": "C"
      },
      {
        "id": 20,
        "question": "一位工程师正在使用卷积神经网络处理一张分辨率为256x256的彩色卫星图像，以识别其中的建筑物。该图像是RGB格式。在网络的第一个卷积层中，他使用了32个尺寸为5x5的卷积核。请问这个卷积层的输出特征图有多少个通道？",
        "type": "multiple_choice",
        "options": [
          "A.3",
          "B.5",
          "C.32",
          "D.256"
        ],
        "answer": "C"
      },
      {
        "id": 21,
        "question": "在一个用于医学影像分析的深度CNN模型中，输入为1280×1280的X光片。网络包含多个卷积层，其中第一层和最后一层都使用了3×3的卷积核。由于网络深度的不同，这两个卷积层的输入特征图分别映射到原始图像的不同范围。关于这两个卷积核实际感受野的说法，正确的是？",
        "type": "multiple_choice",
        "options": [
          "A.两个卷积核的感受野相同，都是3×3像素",
          "B.第一个卷积核的感受野更大，因为它直接处理原始图像",
          "C.最后一个卷积核的感受野更大，因为它叠加了前面所有层的映射",
          "D.感受野大小只与卷积核尺寸有关，与网络深度无关"
        ],
        "answer": "C"
      },
      {
        "id": 22,
        "question": "小王正在设计一个CNN模型来识别MNIST数据集中的手写数字。他在几个卷积层后加入了最大池化层。以下关于池化层在该场景中主要作用的描述中，最准确的是？",
        "type": "multiple_choice",
        "options": [
          "A.显著增加模型的参数数量，使模型能学习更复杂的笔迹特征",
          "B.通过非线性变换让模型能拟合各种奇特的数字写法",
          "C.对特征图进行下采样，在减少计算量的同时，确保数字即使有轻微位置变化也能被正确识别",
          "D.增加特征图的通道数，从而同时检测数字的多个局部特征如拐角、弧线等"
        ],
        "answer": "C"
      },
      {
        "id": 23,
        "question": "在AlexNet模型中，部分MaxPooling层的结构为核尺寸3x3，步长（stride）为2。当一个尺寸为5x5的特征图输入到此池化层时，输出特征图的尺寸是多少？",
        "type": "multiple_choice",
        "options": [
          "A.3x3",
          "B.2x2",
          "C.1x1",
          "D.4x4"
        ],
        "answer": "B"
      },
      {
        "id": 24,
        "question": "VGG网络的核心组件是“VGG块”。下列哪个选项最准确地描述了一个典型的VGG块的结构特点？",
        "type": "multiple_choice",
        "options": [
          "A.一个卷积层后接一个池化层，通过这种交替结构逐步提取特征",
          "B.连续多个3x3卷积层后接一个2x2最大池化层，通过增加深度来提升性能。",
          "C.使用1x1卷积层进行降维，后接多个5x5卷积层以扩大感受野。",
          "D.每个块内使用多种不同尺寸的卷积核并行处理，最后将结果融合。"
        ],
        "answer": "B"
      },
      {
        "id": 25,
        "question": "GoogLeNet（Inception v1）在深度学习领域引人注目，其主要创新在于引入了一种全新的基础构建模块。这个模块的特点是什么？",
        "type": "multiple_choice",
        "options": [
          "A.连续堆叠多个小尺寸卷积核（如3x3）来构建深度网络。",
          "B.通过残差连接结构，将前一层的输出直接跳层传递到后面，以解决梯度消失问题。",
          "C.使用并行结构的多种尺寸卷积核和池化层，并将它们的输出在通道维度上进行拼接。",
          "D.整个网络由一个非常深的卷积序列构成，不使用任何全连接层。"
        ],
        "answer": "C"
      },
      {
        "id": 26,
        "question": "批量归一化是深度学习中的一种重要技术，它在训练过程中对数据进行转换。关于其工作原理和主要作用，下列哪个描述是最准确的？",
        "type": "multiple_choice",
        "options": [
          "A.它对整个训练数据集进行归一化，以加速模型收敛。",
          "B.它在每个训练批次中，对网络层输入数据的每个特征通道进行归一化，并通过可学习的参数进行缩放和偏移，以此缓解内部协变量偏移问题。",
          "C.它对网络的权重参数进行归一化处理，防止权重值变得过大，以此作为主要的正则化手段来防止过拟合。",
          "D.它只在测试阶段使用，对网络的输入图片进行标准化，使其符合预训练模型的输入要求。"
        ],
        "answer": "B"
      },
      {
        "id": 27,
        "question": "ResNet（残差网络）是深度学习发展中的一个里程碑式模型，它通过引入一种特殊的结构，成功解决了极深网络难以训练的问题。这种核心结构及其主要作用是什么？",
        "type": "multiple_choice",
        "options": [
          "A.使用并行排列的不同尺寸卷积核来提取多尺度特征，并通过1x1卷积控制计算量。",
          "B.通过跳跃连接将输入直接传递到后面的层，与经过卷积层的输出相加，使网络能够学习残差映射，缓解梯度消失问题。",
          "C.在每个卷积层后使用批量归一化和ReLU激活函数，并构建了统一的VGG块结构来增加网络深度。",
          "D.完全弃用全连接层，使用全局平均池化层将特征图直接转换为分类向量，大幅减少模型参数。"
        ],
        "answer": "B"
      },
      {
        "id": 28,
        "question": "常用的图像数据增广操作有哪些？",
        "type": "multiple_response",
        "options": [
          "A.图像翻转",
          "B.图像饱和度调整",
          "C.图像随机截取",
          "D.图像亮度调节"
        ],
        "answer": "ABCD"
      },
      {
        "id": 29,
        "question": "你准备使用微调（fine-tuning）方法，将一个在大型图像数据集（如ImageNet）上预训练好的卷积神经网络，应用到你自己拍摄的少量花卉分类图片上。以下哪种做法最符合微调的核心要求？",
        "type": "multiple_choice",
        "options": [
          "A.重新随机初始化网络所有权重，然后用花卉图片从头开始训练",
          "B.保持预训练模型的所有参数不变，仅调整最后分类层的输出节点数以匹配花卉类别数",
          "C.载入预训练模型权重，用较小的学习率在花卉图片上对整个网络进行少量迭代的训练",
          "D.直接使用预训练模型对花卉图片进行预测，不进行任何训练"
        ],
        "answer": "C"
      },
      {
        "id": 30,
        "question": "在微调一个预训练的图像分类模型（如ResNet）用于医学影像分类时，考虑到预训练模型底层学习的是通用特征（如边缘、纹理），而高层学习的是与原始数据集（如ImageNet）更相关的抽象特征，以下哪种微调策略通常最合理？",
        "type": "multiple_choice",
        "options": [
          "A.固定所有层的参数，只重新训练最后的分类层",
          "B.固定高层参数，只更新底层参数",
          "C.固定底层参数，只更新高层和分类层参数",
          "D.不对任何层进行固定，全部参数都参与更新"
        ],
        "answer": "C"
      },
      {
        "id": 31,
        "question": "在RCNN系列目标检测算法中，关于边界框的预测，以下描述最准确的是：",
        "type": "multiple_choice",
        "options": [
          "A.模型直接预测目标物体在图像中的精确边界框坐标。",
          "B.模型首先生成一系列预定义的锚框，然后预测每个锚框的类别。",
          "C.模型首先生成锚框，然后直接预测一个与锚框形状相同的新边界框。",
          "D.模型首先生成锚框，然后预测从锚框到真实边界框的偏移量。"
        ],
        "answer": "D"
      },
      {
        "id": 32,
        "question": "在一个目标检测项目的开发中，你需要根据模型的预测框与真实框的匹配程度来评估检测质量。团队决定，只有当预测框的位置足够准确时，才认为这是一个正确的检测。下列哪种指标最适合用来定量地衡量“位置足够准确”？",
        "type": "multiple_choice",
        "options": [
          "A.分类置信度：即模型预测框内包含目标物体的概率大小。",
          "B.交并比：即预测框与真实框的交集面积与并集面积的比值。",
          "C.像素精度：即预测框内被正确分类的像素点所占的比例。",
          "D.检测速度：即模型处理单张图像并输出所有预测框所需的时间。"
        ],
        "answer": "B"
      },
      {
        "id": 33,
        "question": "你训练了一个目标检测模型来识别图像中的水果。在对一张测试图片进行预测时，模型在同一个苹果周围输出了三个高度重叠的边界框（如下所示），每个框都带有“苹果”标签和不同的置信度（0.9, 0.85, 0.7）。作为后处理步骤，你应该采用下列哪种方法来确保每个苹果只对应一个最准确的边界框？",
        "type": "multiple_choice",
        "options": [
          "A.置信度阈值过滤：设置一个阈值（如0.8），只保留置信度高于此阈值的预测框。",
          "B.非最大抑制：首先选择置信度最高的框，然后抑制掉与其高度重叠且属于同一类别的其他预测框。",
          "C.边界框聚类：将所有预测框的坐标进行平均，生成一个全新的边界框。",
          "D.多尺度融合：将不同尺寸特征图上得到的预测框合并在一起，以捕获更多细节。"
        ],
        "answer": "B"
      },
      {
        "id": 34,
        "question": "你的团队正在开发一个车辆检测系统，需要对图像中上千个候选区域进行识别和定位。初期使用R-CNN模型，发现检测速度极慢，主要瓶颈在于每个候选区域都需要独立通过CNN进行特征提取，计算冗余巨大。为了显著提升系统效率，下列哪种改进策略最符合Fast R-CNN的核心思想？",
        "type": "multiple_choice",
        "options": [
          "A.放弃使用CNN，转而使用更轻量级的传统特征提取方法，如HOG特征。",
          "B.在整个图像上仅执行一次CNN前向传播，得到全局特征图，然后通过“感兴趣区域池化”从该特征图上为每个候选区域提取固定大小的特征。",
          "C.大幅减少生成的候选区域数量，即使这可能牺牲一些对小物体的检测能力。",
          "D.将特征提取和分类任务部署到两个不同的专用硬件上，通过并行计算来减少时间。"
        ],
        "answer": "B"
      },
      {
        "id": 35,
        "question": "你正在为一个自动驾驶的自行车机器人选择目标检测模型。该机器人搭载的处理器计算能力有限，但对实时性要求极高（需要极高的帧率）。在模型评估阶段，你发现Fast R-CNN准确度不错，但速度较慢；而YOLOv1速度极快，但偶尔会将多个小物体检测成一个。从这两种算法的根本性差异来看，造成上述现象的核心原因最可能是：",
        "type": "multiple_choice",
        "options": [
          "A.Fast R-CNN在CPU上运行，而YOLOv1在GPU上运行。",
          "B.Fast R-CNN使用了更深的神经网络 backbone，导致其特征提取能力更强。",
          "C.Fast R-CNN是一个“两阶段”检测器，先生成候选区域再分类；而YOLOv1是一个“单阶段”检测器，将检测视为一个直接的回归问题。",
          "D.YOLOv1在训练时使用的数据集包含的物体类别比Fast R-CNN更少。"
        ],
        "answer": "C"
      },
      {
        "id": 36,
        "question": "在一阶马尔科夫模型中，该假设如何简化序列中状态（如句子中的单词）的依赖关系？",
        "type": "multiple_choice",
        "options": [
          "A.当前状态与所有过去和未来的状态都无关。",
          "B.当前状态的概率仅由其前一个状态决定。",
          "C.当前状态的概率由其前两个状态共同决定。",
          "D.当前状态的概率需要由整个历史序列决定。"
        ],
        "answer": "B"
      },
      {
        "id": 37,
        "question": "关于自回归模型（Autoregressive Model）的核心思想，以下哪一项描述是最准确的？",
        "type": "multiple_choice",
        "options": [
          "A.模型利用外部标签信息来预测序列中的每一个元素。",
          "B.模型使用序列中未来的已知元素来预测过去缺失的元素。",
          "C.模型在预测序列的下一个元素时，依赖于其自身过去已生成的元素。",
          "D.模型同时并行地生成整个序列的所有元素，不依赖历史信息。"
        ],
        "answer": "C"
      },
      {
        "id": 38,
        "question": "在隐马尔可夫模型中，其名称中的“隐”字，指的是什么？",
        "type": "multiple_choice",
        "options": [
          "A.模型的参数是隐藏起来、无法直接学习的。",
          "B.模型的状态转移过程是隐藏的、不可观测的。",
          "C.模型内部的数学计算过程对使用者是隐藏的。",
          "D.模型输出的观测序列是隐藏的、不存在的。"
        ],
        "answer": "B"
      },
      {
        "id": 39,
        "question": "在训练一个现代语言模型之前，为什么通常需要对原始文本语料库进行“分词”操作？",
        "type": "multiple_choice",
        "options": [
          "A.为了将文本翻译成英文，以便模型统一处理。",
          "B.为了将连续的字符序列切分成具有语义的离散基本单元，作为模型的输入。",
          "C.为了删除文本中的所有停用词（如“的”、“了”），只保留关键词。",
          "D.为了自动修正文本中的拼写和语法错误，净化语料。"
        ],
        "answer": "B"
      },
      {
        "id": 40,
        "question": "在循环神经网络中，“隐变量”（或“隐藏状态”）的主要作用是什么？",
        "type": "multiple_choice",
        "options": [
          "A.存储模型最终需要输出的预测结果。",
          "B.存储和传递序列在历史时间步的摘要信息。",
          "C.决定模型参数中哪些部分需要被随机丢弃以防止过拟合。",
          "D.作为一个临时变量，仅在当前时间步的计算中使用，然后被丢弃。"
        ],
        "answer": "B"
      },
      {
        "id": 41,
        "question": "在英译中机器翻译任务中，处理句子 “The cat chased the mouse” 时，自注意力机制与循环神经网络（RNN）的处理方式差异主要体现在？。",
        "type": "multiple_choice",
        "options": [
          "A.自注意力机制只能处理 5 个单词以内的短句，RNN 可处理更长句子",
          "B.自注意力机制能同时 “看到”“cat”“chased”“mouse” 等所有单词的关联，RNN 需从 “The” 开始逐个顺序读取",
          "C.自注意力机制必须先将单词转为独热编码才能计算，RNN 可直接用 Word Embedding",
          "D.自注意力机制无法区分 “cat” 和 “mouse” 的语义，RNN 能通过顺序记忆实现区分"
        ],
        "answer": "B"
      },
      {
        "id": 42,
        "question": "某 NLP 模型分析句子 “小红带了一本书，她在地铁上读完了它” 时，需计算代词 “她” 与句中其他成分的关联性。此时用于表示 “她” 与 “小红” 关联性强弱的符号是？（ ）",
        "type": "multiple_choice",
        "options": [
          "A.α（阿尔法）",
          "B.q（Query，查询向量）",
          "C.k（Key，键向量）",
          "D.v（Value，值向量）"
        ],
        "answer": "D"
      },
      {
        "id": 43,
        "question": "在 “中文句子‘人工智能很重要’→英文翻译‘Artificial intelligence is important’” 的任务中，Transformer 解码器生成 “intelligence” 一词时，调用的交叉注意力模块中，Query（q）、Key（k）、Value（v）的来源是？（ ）",
        "type": "multiple_choice",
        "options": [
          "A.q、k、v 均来自 “人工智能很重要” 的编码器输出",
          "B.q 来自解码器已生成的 “Artificial” 向量，k、v 来自编码器输出",
          "C.q、k、v 均来自解码器已生成的 “Artificial” 向量",
          "D.q 来自编码器输出，k、v 来自解码器已生成的 “Artificial” 向量"
        ],
        "answer": "A"
      },
      {
        "id": 44,
        "question": "某图像处理模型需识别图片中 “小孩抱着玩具熊” 的细节，对比 CNN 与自注意力机制的处理逻辑，下列描述符合实际场景的是？（ ）",
        "type": "multiple_choice",
        "options": [
          "A.CNN 能直接关注 “小孩的手” 与 “玩具熊” 的全局关联，自注意力机制只能看局部像素",
          "B.自注意力机制可直接捕捉 “小孩” 与 “玩具熊” 的整体位置关系，CNN 需通过多层卷积堆叠才能扩大感受野",
          "C.CNN 的卷积核参数是动态学习的，自注意力机制的注意力矩阵是固定不变的",
          "D.两者无法通过调整参数适配该图像识别场景"
        ],
        "answer": "B"
      },
      {
        "id": 45,
        "question": "某语音助手需处理用户一段 15 秒的语音指令（语音信号每秒约 100 帧，共 1500 帧），为避免注意力计算量过大，模型最可能采用的优化方式是？（ ）",
        "type": "multiple_choice",
        "options": [
          "A.直接生成 1500×1500 的完整注意力矩阵，保证计算精度",
          "B.使用 “截断注意力（Truncated Self-attention）”，仅让注意力关注相邻的 200 帧区域",
          "C.取消位置编码，减少 1500 帧的序列信息冗余",
          "D.用传统 RNN 替代自注意力机制，完全规避矩阵计算"
        ],
        "answer": "B"
      },
      {
        "id": 46,
        "question": "关于Diffusion模型的基本过程，下列哪一项描述是正确的？",
        "type": "multiple_choice",
        "options": [
          "A.前向加噪过程（扩散过程）中，模型逐步学习从噪声中恢复原始图像",
          "B.模型训练阶段的目标是学习如何向图像中添加高斯噪声",
          "C.前向加噪过程通过逐步添加噪声将图像破坏为纯噪声，模型训练则学习从噪声中重建图像",
          "D.模型训练完成后，直接通过一次前向传播即可从噪声生成清晰图像"
        ],
        "answer": "C"
      },
      {
        "id": 47,
        "question": "在条件Diffusion模型（Text-to-Image）的训练过程中，关于文本条件输入的使用方式，下列哪一项是正确的？",
        "type": "multiple_choice",
        "options": [
          "A.模型在每一步去噪训练时，会接收不同的文本描述作为条件输入",
          "B.模型仅在训练开始时接收一次文本条件，后续去噪步骤不再使用",
          "C.模型在每一步去噪训练时，都会接收同一个文本描述作为条件输入",
          "D.文本条件仅在推理阶段使用，训练时模型完全不需要文本输入"
        ],
        "answer": "C"
      },
      {
        "id": 48,
        "question": "关于Diffusion模型中FID（Fréchet Inception Distance）指标的说法，下列哪一项是正确的？",
        "type": "multiple_choice",
        "options": [
          "A.FID计算的是单张生成图像与单张真实图像之间的像素级差异",
          "B.FID值越大，表明生成图像的质量和多样性越好",
          "C.FID通过比较生成图像集和真实图像集在特征空间中的分布距离来评估模型性能",
          "D.FID主要衡量的是模型生成图像的速度，数值越小代表生成效率越高"
        ],
        "answer": "C"
      },
      {
        "id": 49,
        "question": "关于Stable Diffusion模型，下列哪一项描述是正确的？",
        "type": "multiple_choice",
        "options": [
          "A.Stable Diffusion直接在原始像素空间进行所有扩散和去噪操作",
          "B.Stable Diffusion通过编码器将图像压缩到潜空间，并在潜空间完成扩散过程的核心操作",
          "C.Stable Diffusion的“压缩”步骤只是为了减少存储占用，对生成质量没有实质性影响",
          "D.Stable Diffusion在潜空间进行扩散，但“理解”图像的步骤只在解码器输出时才发生"
        ],
        "answer": "B"
      },
      {
        "id": 50,
        "question": "在训练Stable Diffusion（文本到图像生成）模型时，关于噪声添加的位置，下列哪一项是正确的？",
        "type": "multiple_choice",
        "options": [
          "A.随机噪声直接添加到原始RGB图像的像素上",
          "B.随机噪声添加到从图像通过编码器得到的潜空间表示上",
          "C.随机噪声同时添加到原始图像和文本嵌入向量上",
          "D.随机噪声只添加到UNet模型的中间层特征上"
        ],
        "answer": "B"
      }
    ]
  },
  {
    "id": "javaee",
    "name": "JavaEE (Spring/Boot/MVC)",
    "questions": [
      {
        "id": 1,
        "question": "Spring是一个轻量级的Java开发框架",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 2,
        "question": "注入bean的注解中，默认按照Bean类型进行装备的注解是@Resource",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 3,
        "question": "Spring框架实例化Bean最常用的方法是构造方法实例化。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 4,
        "question": "Spring的核心容器包含的模块有",
        "type": "multiple_response",
        "options": [
          "A. Spring-context",
          "B. Spring-expression",
          "C. Spring-context-support",
          "D. Spring-core",
          "E. Spring-beans"
        ],
        "answer": "ABCDE",
        "raw_answer": "ABCDE:Spring-context; Spring-expression; Spring-context-support; Spring-core; Spring-beans;"
      },
      {
        "id": 5,
        "question": "关于Spring的IOC，以下说法正确的是",
        "type": "multiple_response",
        "options": [
          "A. 对象之间的关系有Spring容器进行管理控制",
          "B. Spring容器负责将被依赖对象赋值给调用者的成员变量。",
          "C. Spring容器负责创建对象",
          "D. 依赖注入的目的是为解耦"
        ],
        "answer": "BD",
        "raw_answer": "BD:Spring容器负责将被依赖对象赋值给调用者的成员变量。; 依赖注入的目的是为解耦;"
      },
      {
        "id": 6,
        "question": "Spring框架实例化Bean的方法有",
        "type": "multiple_response",
        "options": [
          "A. 构造方法实例化",
          "B. 静态工厂实例化",
          "C. 实例工厂实例化",
          "D. 初始化方法实例化"
        ],
        "answer": "ABC",
        "raw_answer": "ABC:构造方法实例化; 静态工厂实例化; 实例工厂实例化;"
      },
      {
        "id": 7,
        "question": "Spring中提供IOC/DI功能的模块是",
        "type": "multiple_choice",
        "options": [
          "A. Spring-context",
          "B. Spring-expression",
          "C. Spring-context-support",
          "D. Spring-beans",
          "E. Spring-core"
        ],
        "answer": "E",
        "raw_answer": "E:Spring-core;"
      },
      {
        "id": 8,
        "question": "Spring中提供对象管理的模块是",
        "type": "multiple_choice",
        "options": [
          "A. Spring-beans",
          "B. Spring-expression",
          "C. Spring-context-support",
          "D. Spring-context",
          "E. Spring-core"
        ],
        "answer": "A",
        "raw_answer": "A:Spring-beans;"
      },
      {
        "id": 9,
        "question": "SpringMVC中寻找控制器的组件是",
        "type": "multiple_choice",
        "options": [
          "A. Controller",
          "B. HandlerMapping",
          "C. DispatcherServlet",
          "D. servlet-name-servlet.xml"
        ],
        "answer": "B",
        "raw_answer": "B:HandlerMapping;"
      },
      {
        "id": 10,
        "question": "小明的SpringMVC应用的包com.example.demo下创建了该应用的Java配置类，以下哪个组合是正确的",
        "type": "multiple_choice",
        "options": [
          "A.",
          "B.",
          "C.",
          "D."
        ],
        "answer": "B",
        "raw_answer": "B:Java配置类中的扫描注解为：@ComponentScan 控制器类所在包为com.example.demo.controller;"
      },
      {
        "id": 11,
        "question": "\n有如下控制器类\n@RequestMapping(\"/user\")\npublic class TestController(){\n@RequestMapping(\"/findByName\")\npublic String addUser(User user, Model){ .....}\n}\n如果该应用的上下文路径（context-path）为/myoa，则下列哪个请求会执行addUser方法",
        "type": "multiple_choice",
        "options": [
          "A. http://localhost:8080/user/findByName",
          "B. http://localhost:8080/myoa/user/",
          "C. http://localhost:8080/myoa/user/findByName",
          "D. http://localhost:8080/myoa/findByName"
        ],
        "answer": "C",
        "raw_answer": "C:http://localhost:8080/myoa/user/findByName;"
      },
      {
        "id": 12,
        "question": "作为控制器方法参数的Model对象model，其存在的生命周期是",
        "type": "multiple_choice",
        "options": [
          "A. session",
          "B. page",
          "C. request",
          "D. application"
        ],
        "answer": "C",
        "raw_answer": "C:request;"
      },
      {
        "id": 13,
        "question": "SpringMVC的核心控制器是",
        "type": "multiple_choice",
        "options": [
          "A. Listener",
          "B. Filter",
          "C. Servlet",
          "D. JSP"
        ],
        "answer": "C",
        "raw_answer": "C:Servlet;"
      },
      {
        "id": 14,
        "question": "对于SpringMVC的Java配置文件，可以实现的接口是",
        "type": "multiple_choice",
        "options": [
          "A. WebMvcConfigurer",
          "B. Configuration",
          "C. WebApplicationInitializer",
          "D. AnnotationConfigWebApplicationContext"
        ],
        "answer": "A",
        "raw_answer": "A:WebMvcConfigurer;"
      },
      {
        "id": 15,
        "question": "在SpringMVC中，负责查找View并将模型数据传给视图，从而将结果渲染给客户端的是组件是",
        "type": "multiple_choice",
        "options": [
          "A. View(JSP、HTML等)",
          "B. ModelAndView",
          "C. Model",
          "D. ViewResolver"
        ],
        "answer": "D",
        "raw_answer": "D:ViewResolver;"
      },
      {
        "id": 16,
        "question": "SpringMVC中，对HTTP请求进行实际处理的是",
        "type": "multiple_choice",
        "options": [
          "A. HandlerMapping",
          "B. ViewResolver",
          "C. DispatcherServlet",
          "D. Controller"
        ],
        "answer": "D",
        "raw_answer": "D:Controller;"
      },
      {
        "id": 17,
        "question": "在SpringMVC的web应用程序中 ，如果想使用Java配置文件对DispatchServlet进行配置，则可以实现的接口或继承的类有",
        "type": "multiple_response",
        "options": [
          "A. WebApplicationInitializer",
          "B. AnnotationConfigWebApplicationContext",
          "C. WebMvcConfigurer",
          "D. AbstractAnnotationConfigDispatcherServletInitializer"
        ],
        "answer": "AD",
        "raw_answer": "AD:WebApplicationInitializer; AbstractAnnotationConfigDispatcherServletInitializer;"
      },
      {
        "id": 18,
        "question": "SpringMVC中属于控制器的有",
        "type": "multiple_response",
        "options": [
          "A. 各个Controller",
          "B. HandlerMapping",
          "C. JSP、HTML",
          "D. DispatcherServlet"
        ],
        "answer": "ABD",
        "raw_answer": "ABD:各个Controller; HandlerMapping; DispatcherServlet;"
      },
      {
        "id": 19,
        "question": "关于SpringMVC的Java配置文件中的注解，下列描述正确的是",
        "type": "multiple_response",
        "options": [
          "A. 需要使用@Bean注解配置视图解析器、格式转换器等组件。",
          "B. 需要使用@EnableWebMvc注解开启默认配置",
          "C. 需使用@Configuration注解声明该类为Java配置类。",
          "D. 需要使用@ComponentScan注解扫描应用所需的各种用注解生成和注入的类"
        ],
        "answer": "ABCD",
        "raw_answer": "ABCD:需要使用@Bean注解配置视图解析器、格式转换器等组件。; 需要使用@EnableWebMvc注解开启默认配置; 需使用@Configuration注解声明该类为Java配置类。; 需要使用@ComponentScan注解扫描应用所需的各种用注解生成和注入的类;"
      },
      {
        "id": 20,
        "question": "\n想让参数book暴露为模型数据，且其名称为bookInfo，则以下控制器方法正确的有，",
        "type": "multiple_response",
        "options": [
          "A.",
          "B.",
          "C.",
          "D."
        ],
        "answer": "BC",
        "raw_answer": "BC:@RequestMapping(\"/add\") public String addBook(Book book, Model model){ ........ model.addAttribute(\"bookInfo\", book) ........ } ; @RequestMapping(\"/add\") public String addBook(@ModelAttribute(\"bookinfo\") Book book){.........} ;"
      },
      {
        "id": 21,
        "question": "控制器如果使用实体bean接受请求参数，则bean的属性名称必须和请求参数名称相同。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 22,
        "question": "如果想使用处理方法的形参接受请求参数，则必须在形参前加上@RequestParam注解",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 23,
        "question": "使用处理方法的形参接收请求参数，则要求形参名称和请求参数名称必须一致。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 24,
        "question": "SpringMVC控制器处理方法中的return语句默认实现的重定向到视图。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 25,
        "question": "在SpringMVC应用程序中所有请求都应该由DispatcherServlet进行处理。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 26,
        "question": "使用注解@RequestParam接受请求参数，如果处理方法的形参名称和请求参数名称不一致，则不会报400错误。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 27,
        "question": "SpringMVC应用中，数据访问层是模型（Model）的一部分",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 28,
        "question": "SpringMVC应用中视图层（view）只能使用JSP",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 29,
        "question": "\n下列页面定了四则运算的表单：\n<form action=\"_____________\"  method=\"post\">\n<input type=\"text\" name=\"op1\" value=\"${equation.op1}\">\n<select name=\"opr\">\n<option value=\"+\" ${equation.opr == '+' ? 'selected' : ''}>+</option>\n<option value=\"-\"  ${equation.opr == '-' ? 'selected' : ''}>-</option>\n<option value=\"*\"  ${equation.opr == '*' ? 'selected' : ''}>*</option>\n<option value=\"/\"  ${equation.opr == '/' ? 'selected' : ''}>/</option>\n</select>\n<input type=\"text\" name=\"op2\" value=\"${equation.op2}\">\n<input type=\"submit\" value=\"=\">\n<input type=\"text\" name=\"result\" value=\"${equation.result}\" disabled>\n</form>\n该表单对应的Controller中定了响应上述表单的端点：\n@Controller\n@Slf4j\n@RequestMapping(\"/arith\")\npublic class ArithmeticController {\nprivate CalculateService calculateService;\n@Autowired\npublic ArithmeticController(CalculateService calculateService) {\nthis.calculateService = calculateService;\n}\n@GetMapping(\"/\")\npublic String index() {\nreturn \"arithmetic\";\n}\n@PostMapping(\"/calcu\")\npublic String calcu(@ModelAttribute Equation equation) {\nlog.debug(\"算式: {} {} {}\",equation.getOp1(),equation.getOpr(),equation.getOp2());\ncalculateService.calculate(equation);\nreturn \"arithmetic\";\n}\n}\n如果程序的context-path为math，则表单actio属性处应该填写",
        "type": "multiple_choice",
        "options": [
          "A. /math/arith",
          "B. /arith/calcu",
          "C. /math/calcu",
          "D. /math/arith/calcu"
        ],
        "answer": "D",
        "raw_answer": "D:/math/arith/calcu;"
      },
      {
        "id": 30,
        "question": "\nSpring MVC程序的配置类中有如下配置：\n@Configuration\n@EnableWebMvc\n@ComponentScan(basePackages ={\"controller\",\"service\"})\npublic class SpringMVCConfig implements WebMvcConfigurer {\n@Override\npublic void addResourceHandlers(ResourceHandlerRegistry registry) {\nregistry.addResourceHandler(\"/image/\").addResourceLocations(\"/img/\");\nregistry.addResourceHandler(\"/css/\").addResourceLocations(\"/css/\");\nregistry.addResourceHandler(\"/js/\").addResourceLocations(\"/js/\");\n}\n}\n如果程序的context-path为test，在img目录下有rose.jpg图片，为确保能获取该图片，则页面的img元素该如何编写(该页面的实际的访问路径未知)：\n<img src=\"___________\" >",
        "type": "multiple_choice",
        "options": [
          "A. image/rose.jpg",
          "B. img/rose.jpg",
          "C. /test/img/rose.jpg",
          "D. /test/image/rose.jpg",
          "E. /image/rose.jpg",
          "F. /img/rose.jpg"
        ],
        "answer": "D",
        "raw_answer": "D:/test/image/rose.jpg;"
      },
      {
        "id": 31,
        "question": "关于@Autowired注解实现依赖注入的说法错误的是",
        "type": "multiple_choice",
        "options": [
          "A. @Autowired可以标注在需要注入的属性上",
          "B. @Autowired可以标注在需要注入对象的getter方法上",
          "C. @Autowired可以标注在用于初始化注入对象的构造方法上",
          "D. @Autowired可以标注在需要注入对象的setter方法上"
        ],
        "answer": "B",
        "raw_answer": "B:@Autowired可以标注在需要注入对象的getter方法上;"
      },
      {
        "id": 32,
        "question": "\n访问arithmetic页面的端点定义在下面代码中：\n@Controller\n@RequestMapping(\"/arith\")\npublic class ArithmeticController {\nprivate CalculateService calculateService;\n@Autowired\npublic ArithmeticController(CalculateService calculateService) {\nthis.calculateService = calculateService;\n}\n@GetMapping(\"/\")\npublic String index() {\nreturn \"arithmetic\";\n}\n}\n项目的context-path为math，端口为8080，则访问arithmetic页面的URL为：",
        "type": "multiple_choice",
        "options": [
          "A. http://xxxxxxxx:8080/arith/math",
          "B. http://xxxxxxxx:8080/arith/",
          "C. http://xxxxxxxx:8080/math/arith/",
          "D. http://xxxxxxxx:8080/math/arith"
        ],
        "answer": "C",
        "raw_answer": "C:http://xxxxxxxx:8080/math/arith/;"
      },
      {
        "id": 33,
        "question": "\n接受表单请求的Controller方法如下：\n@PostMapping(\"/calcu\")\npublic String calcu(@ModelAttribute Equation equation) {\nlog.debug(\"算式: {} {} {}\",equation.getOp1(),equation.getOpr(),equation.getOp2());\ncalculateService.calculate(equation);\nreturn \"arithmetic\";\n}\n以下说法正确的是：",
        "type": "multiple_choice",
        "options": [
          "A. 此处@ModelAttribute的作用是确保执行指定的方法",
          "B. 此处@ModelAttribute的作用可有可无",
          "C. @ModelAttribute的作用是确保形参equation能接受到参数，如果equation的属性名与请求名不一致，会报400错误",
          "D."
        ],
        "answer": "D",
        "raw_answer": "D:@ModelAttribute的作用是将形参暴露为模型，保存在model中，供arithmetic组件使用 ;"
      },
      {
        "id": 34,
        "question": "SpringMVC控制器处理方法中的return语句默认实现的重定向到视图。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 35,
        "question": "重定向和转发一样，只存在一组request/response对象",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 36,
        "question": "如果一个请求需要由多个组件完成处理，则组件之间应该采用重定向跳转。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 37,
        "question": "Spring Boot的应用程序一般打包为",
        "type": "multiple_choice",
        "options": [
          "A. .jar",
          "B. .war"
        ],
        "answer": "A",
        "raw_answer": "A:.jar;"
      },
      {
        "id": 38,
        "question": "如果想改变SpringBoot的默认配置，可选的、合适的途径有",
        "type": "multiple_response",
        "options": [
          "A. 在全局配置文件中修改默认属性",
          "B. 重新定义或声明某些默认的Bean",
          "C. 创建和声明自定义的Bean",
          "D. 修改SpringBoot 源代码后重新编译构建"
        ],
        "answer": "AB",
        "raw_answer": "AB:在全局配置文件中修改默认属性; 重新定义或声明某些默认的Bean;"
      },
      {
        "id": 39,
        "question": "以下哪些注解或对象可以读取全局配置文件application.properties的信息",
        "type": "multiple_response",
        "options": [
          "A. Environment 对象",
          "B. @ConfigurationProperties",
          "C. @PropertySource",
          "D. @Value"
        ],
        "answer": "ABD",
        "raw_answer": "ABD:Environment 对象; @ConfigurationProperties; @Value;"
      },
      {
        "id": 40,
        "question": "可以作为springboot应用程序全局配置文件的有",
        "type": "multiple_response",
        "options": [
          "A. application.yml",
          "B. application.config",
          "C. application.yaml",
          "D. application.properties"
        ],
        "answer": "ACD",
        "raw_answer": "ACD:application.yml; application.yaml; application.properties;"
      },
      {
        "id": 41,
        "question": "SpringBoot应用程序，需要自己单独配置Web容器。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 42,
        "question": "SpringBoot的全局配置文件只能修改项目的默认配置，不能添加项目特有的配置。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 43,
        "question": "application.properties和application.yml不能共存。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 44,
        "question": "Spring Boot应用程序应该需要配置Tomcat等容器。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 45,
        "question": "Spring Boot应用程序是从main方法开始运行的。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 46,
        "question": "SpringBoot使用MAVEN等构件工具构件应用程序。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 47,
        "question": "SpringBoot是完全不同于Spring框架的全新框架。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 48,
        "question": "小明的SpringMVC应用的包com.example.demo下创建了该应用的Java配置类，以下哪个组合是正确的",
        "type": "multiple_choice",
        "options": [
          "A.",
          "B.",
          "C.",
          "D."
        ],
        "answer": "D",
        "raw_answer": "D:Java配置类中的扫描注解为：@ComponentScan 控制器类所在包为com.example.demo.controller;"
      },
      {
        "id": 49,
        "question": "SpringBoot应用中，如果@SpringBootApplication注解的入口类放置在项目的com.example.hello包下，则其他的配置类和用注解声明的bean类应该放在",
        "type": "multiple_choice",
        "options": [
          "A. 可以放在com.example包下",
          "B. 放在com.example.hello包及其子包下",
          "C. 只能放在com.example.hello包下",
          "D. 可以放在任意包下"
        ],
        "answer": "B",
        "raw_answer": "B:放在com.example.hello包及其子包下;"
      },
      {
        "id": 50,
        "question": "作为控制器方法参数的Model对象model，其存在的生命周期是",
        "type": "multiple_choice",
        "options": [
          "A. application",
          "B. page",
          "C. request",
          "D. session"
        ],
        "answer": "C",
        "raw_answer": "C:request;"
      },
      {
        "id": 51,
        "question": "SpringMVC的核心控制器是",
        "type": "multiple_choice",
        "options": [
          "A. Filter",
          "B. JSP",
          "C. Servlet",
          "D. Listener"
        ],
        "answer": "C",
        "raw_answer": "C:Servlet;"
      },
      {
        "id": 52,
        "question": "Spring Boot的应用程序一般打包为",
        "type": "multiple_choice",
        "options": [
          "A. .jar",
          "B. .war"
        ],
        "answer": "A",
        "raw_answer": "A:.jar;"
      },
      {
        "id": 53,
        "question": "\n有如下控制器类\n@RequestMapping(\"/user\")\npublic class TestController(){\n@RequestMapping(\"/findByName\")\npublic String addUser(User user, Model){ .....}\n}\n如果该应用的上下文路径（context-path）为/myoa，则下列哪个请求会执行addUser方法",
        "type": "multiple_choice",
        "options": [
          "A. http://localhost:8080/myoa/user/findByName",
          "B. http://localhost:8080/user/findByName",
          "C. http://localhost:8080/myoa/user/",
          "D. http://localhost:8080/myoa/findByName"
        ],
        "answer": "A",
        "raw_answer": "A:http://localhost:8080/myoa/user/findByName;"
      },
      {
        "id": 54,
        "question": "Thymeleaf的动态模板功能通常是放置在HTML标签的___________上实现的。",
        "type": "multiple_choice",
        "options": [
          "A. 标签内容",
          "B. 使用动态标签库",
          "C. 属性",
          "D. 标签名"
        ],
        "answer": "C",
        "raw_answer": "C:属性;"
      },
      {
        "id": 55,
        "question": "SpringBoot使用MAVEN等构件工具构件应用程序。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 56,
        "question": "控制器如果使用实体bean接受请求参数，则bean的属性名称必须和请求参数名称相同。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 57,
        "question": "Spring Boot应用程序是从main方法开始运行的。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 58,
        "question": "Spring Boot应用程序应该需要配置Tomcat等容器。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 59,
        "question": "使用注解@RequestParam接受请求参数，如果处理方法的形参名称和请求参数名称不一致，则不会报400错误。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 60,
        "question": "SpringBoot是完全不同于Spring框架的全新框架。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 61,
        "question": "SpringBoot的全局配置文件只能修改项目的默认配置，不能添加项目特有的配置。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 62,
        "question": "在SpringMVC应用程序中所有请求都应该由DispatcherServlet进行处理。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 63,
        "question": "有表单<form th:action=\"@{/login}\" th:object=\"${user}\">...</form>，这里的user对象不必一定存在。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 64,
        "question": "\n想让参数book暴露为模型数据，且其名称为bookInfo，则以下控制器方法正确的有，",
        "type": "multiple_response",
        "options": [
          "A.",
          "B.",
          "C.",
          "D."
        ],
        "answer": "AB",
        "raw_answer": "AB:@RequestMapping(\"/add\") public String addBook(@ModelAttribute(\"bookinfo\") Book book){.........} ; @RequestMapping(\"/add\") public String addBook(Book book, Model model){ ........ model.addAttribute(\"bookInfo\", book) ........ } ;"
      },
      {
        "id": 65,
        "question": "可以作为springboot应用程序全局配置文件的有",
        "type": "multiple_response",
        "options": [
          "A. application.yml",
          "B. application.config",
          "C. application.properties",
          "D. application.yaml"
        ],
        "answer": "ACD",
        "raw_answer": "ACD:application.yml; application.properties; application.yaml;"
      },
      {
        "id": 66,
        "question": "有表单<form th:action=\"@{/login}\">...</form>，当前项目上下文路径(context-path)为myoa，则转换成html后，action的属性值应该是哪一个。（至少选一项）",
        "type": "multiple_response",
        "options": [
          "A. /myoa/login",
          "B. /login",
          "C. login",
          "D. myoa/login"
        ],
        "answer": "A",
        "raw_answer": "A:/myoa/login;"
      },
      {
        "id": 67,
        "question": "SpringMVC中属于控制器的有",
        "type": "multiple_response",
        "options": [
          "A. JSP、HTML",
          "B. HandlerMapping",
          "C. DispatcherServlet",
          "D. 各个Controller"
        ],
        "answer": "BCD",
        "raw_answer": "BCD:HandlerMapping; DispatcherServlet; 各个Controller;"
      },
      {
        "id": 68,
        "question": "关于SpringMVC文件上传，说法正确的是",
        "type": "multiple_response",
        "options": [
          "A. 需要在SpringMVC应用的配置中配置MultipartResolver",
          "B. 传页面表单中的enctype属性必须为multipart/form-data",
          "C. 上传页面表单中的method属性必须为post",
          "D. 控制器方法参数中需有一个类型为MultipartFile的参数"
        ],
        "answer": "ABCD",
        "raw_answer": "ABCD:需要在SpringMVC应用的配置中配置MultipartResolver; 传页面表单中的enctype属性必须为multipart/form-data; 上传页面表单中的method属性必须为post; 控制器方法参数中需有一个类型为MultipartFile的参数;"
      },
      {
        "id": 69,
        "question": "以下哪些是正确的JSON对象",
        "type": "multiple_response",
        "options": [
          "A. [\"足球\",\"篮球\",\"排球\"]",
          "B.",
          "C. [\"张三\",  {\"name\":\"李四\", \"age\":20}, [\"王五\", \"马六\"]]",
          "D. {\"tname\":\"张三\", \"tid\":1001}"
        ],
        "answer": "ABCD",
        "raw_answer": "ABCD:[\"足球\",\"篮球\",\"排球\"]; { \"name\":\"jack\", \"hobbys\":[\"旅游\",\"美食\",\"电游\"], \"pet\": {\"type\":\"dog\", \"name\":\"wangwang\"} }; [\"张三\", {\"name\":\"李四\", \"age\":20}, [\"王五\", \"马六\"]]; {\"tname\":\"张三\", \"tid\":1001};"
      },
      {
        "id": 70,
        "question": "@RestController相当于哪几个注解的组合",
        "type": "multiple_response",
        "options": [
          "A. @RequestBody",
          "B. @Service",
          "C. @Controller",
          "D. @ResponseBody"
        ],
        "answer": "CD",
        "raw_answer": "CD:@Controller; @ResponseBody;"
      },
      {
        "id": 71,
        "question": "JSON数据结构中值（value）的数据类型可以是",
        "type": "multiple_response",
        "options": [
          "A. Object",
          "B. true/false",
          "C. Number",
          "D. Array",
          "E. String"
        ],
        "answer": "ABCDE",
        "raw_answer": "ABCDE:Object; true/false; Number; Array; String;"
      },
      {
        "id": 72,
        "question": "JSON的对象结构中，其key的数据类型必须为String",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 73,
        "question": "JSON的两种数据结构可以嵌套或组合，构成更为复杂的结构",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 74,
        "question": "SpringMVC实现浏览器和控制器处理方法间的JSON数据交互时，同时使用了@RequestBody和@ResponseBody注解，则处理器方法的return语句中返回的应该是一个Java对象。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 75,
        "question": "SpringMVC实现浏览器和控制器处理方法间的JSON数据交互时，注解@RequestBody使用在处理方法上。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 76,
        "question": "SpringMVC实现浏览器和控制器处理方法间的JSON数据交互时，使用了@ResponseBody注解，则处理方法返回的是视图。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 77,
        "question": "JSON是基于纯文本的数据格式",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 78,
        "question": "文件下载前需要对文件名进行编码，比如保存文件名的变量为filename，需要filename=URLEncoder.encode(filename,\"UTF-8\")转换为才能确保文件能正常下载，其理由是：",
        "type": "multiple_choice",
        "options": [
          "A. 确保文件路径正确",
          "B. 需要对文件名进行加密",
          "C. 原文件名的编码不是UTF-8，需要转换为UTF-8编码格式",
          "D. 文件名中可能包含空格等不能在URL出现的字符，需要对这些字符进行转码（编码）以符合URL的规则要求"
        ],
        "answer": "D",
        "raw_answer": "D:文件名中可能包含空格等不能在URL出现的字符，需要对这些字符进行转码（编码）以符合URL的规则要求;"
      },
      {
        "id": 79,
        "question": "在文件下载时，响应头（header）的Content-Type属性应该是",
        "type": "multiple_choice",
        "options": [
          "A. application/octet-stream",
          "B. application/xhtml+xml",
          "C. application/json",
          "D. text/html",
          "E. application/stream+json"
        ],
        "answer": "A",
        "raw_answer": "A:application/octet-stream;"
      },
      {
        "id": 80,
        "question": "文件下载时，需要设置响应的实体头（header）的哪个属性，用于控制文件下载的行为",
        "type": "multiple_choice",
        "options": [
          "A. Content-Disposition",
          "B. Content-Encoding",
          "C. Content-Type",
          "D. Content-Length"
        ],
        "answer": "A",
        "raw_answer": "A:Content-Disposition;"
      },
      {
        "id": 81,
        "question": "可以使用Map类型封装查询结果集",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 82,
        "question": "MyBatis映射文件中<mappers>元素是配置文件的根元素，它包含一个namespace属性，该属性为这个<mappers>指定了唯一的命名空间。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 83,
        "question": "映射接口文件和映射SQL文件必须在同一个包路径下。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 84,
        "question": "各类SQL元素的resultType和parameterType很多时候可以忽略不写，因为可以自动推断",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 85,
        "question": "动态SQL可以根据参入的参数的具体情况动态的拼接SQL语句",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 86,
        "question": "\nSpringBoot全局文件中配置MyBatis的配置项mybatis.type-aliases-package的作用是给实体类取别名，其值是实体类的别名。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 87,
        "question": "在MyBatis中SQL注解和XML映射文件不能同时共存。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 88,
        "question": "在select元素将查询的结果集封装到resultType指定的实体类的对象中，它是无条件的，可以任意定义实体类",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 89,
        "question": "如果实体类和对应的数据库表的列名不一致，唯一的解决办法就是使用ReusltMap定义类到表的映射",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 90,
        "question": "可以使用Map类型作为参数传递查询条件",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 91,
        "question": "使用mybaits-spring整合MyBatis与Spring时，或者MyBatis集成在SpringBoot中时，当映射器接口和SQL映射文件在同一个包路径下时，不需要再额外指定SQL映射文件的位置了。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 92,
        "question": "定好的resultMap不可以重复使用",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "B",
        "raw_answer": "错"
      },
      {
        "id": 93,
        "question": "当表中的列名与对应对象的属性名称完全一致时，在查询映射文件中可以不使用resultMap属性。",
        "type": "multiple_choice",
        "options": [
          "A. 对",
          "B. 错"
        ],
        "answer": "A",
        "raw_answer": "对"
      },
      {
        "id": 94,
        "question": "MyBatis的SQL映射文件中，以下属于常见元素的是",
        "type": "multiple_response",
        "options": [
          "A. <resultMap>",
          "B. <select>",
          "C. <update>",
          "D. <insert>",
          "E. <sql>",
          "F. <delete>"
        ],
        "answer": "ABCDEF",
        "raw_answer": "ABCDEF:<resultMap>; <select>; <update>; <insert>; <sql>; <delete>;"
      },
      {
        "id": 95,
        "question": "下列选项中，对使用MyBatis编程的好处说法正确的是（）。",
        "type": "multiple_response",
        "options": [
          "A. 自动将Java对象映射至 SQL语句。",
          "B. 实现了SQL与Java 代码的分离。",
          "C. 不用配置数据连接池，也可以高效的管理数据库连接。",
          "D. 自动将SQL执行结果映射至Java对象。"
        ],
        "answer": "ABD",
        "raw_answer": "ABD:自动将Java对象映射至 SQL语句。; 实现了SQL与Java 代码的分离。; 自动将SQL执行结果映射至Java对象。;"
      },
      {
        "id": 96,
        "question": "下面关于映射文件中的<mapper>元素的属性，说法正确的是（）。",
        "type": "multiple_choice",
        "options": [
          "A. resultType属性的值表示传入的参数类型",
          "B. 以上说法都不正确",
          "C. parameterType属性的值表示的是返回的实体类对象",
          "D. namespace属性的值通常设置为对应实体类的全限定类名"
        ],
        "answer": "D",
        "raw_answer": "D:namespace属性的值通常设置为对应实体类的全限定类名;"
      },
      {
        "id": 97,
        "question": "以下关于<select>元素及其属性说法错误的是（）。",
        "type": "multiple_choice",
        "options": [
          "A. resultMap表示外部resultMap的命名引用，返回时可以同时使用resultType和resultMap",
          "B. 在同一个映射文件中可以配置多个<select>元素",
          "C. <select>元素用来映射查询语句，它可以帮助我们从数据库中读取出数据，并组装数据给业务开发人员",
          "D. parameterType属性表示传入SQL语句的参数类的全限定名或者别名"
        ],
        "answer": "A",
        "raw_answer": "A:resultMap表示外部resultMap的命名引用，返回时可以同时使用resultType和resultMap;"
      },
      {
        "id": 98,
        "question": "下列关于<mapper>元素的说法正确的是（）。",
        "type": "multiple_choice",
        "options": [
          "A. <mapper>元素不是映射文件的根元素",
          "B. <mapper>元素是映射文件的根元素",
          "C. <mapper>元素的namespace属性是不唯一的",
          "D. <mapper>元素的namespace属性值的命名不一定跟接口同名"
        ],
        "answer": "B",
        "raw_answer": "B:<mapper>元素是映射文件的根元素;"
      },
      {
        "id": 99,
        "question": "以下有关MyBatis映射文件中<insert>元素说法正确的是（）。",
        "type": "multiple_choice",
        "options": [
          "A. <insert>元素的属性与<select>元素的属性相同",
          "B. <insert>元素用于映射插入语句，在执行完元素中定义的SQL语句后，没有返回结果",
          "C. keyColumn属性用于设置第几列是主键，当主键列不是表中的第一列时需要设置",
          "D. useGeneratedKeys（仅对insert有用）此属性会使MyBatis使用JDBC的getGeneratedKeys()方法来获取由数据库内部生产的主键"
        ],
        "answer": "C",
        "raw_answer": "C:keyColumn属性用于设置第几列是主键，当主键列不是表中的第一列时需要设置;"
      },
      {
        "id": 100,
        "question": "下列关于MyBatis中默认的常见Java类型的别名，正确的是（）。",
        "type": "multiple_choice",
        "options": [
          "A. 映射类型为Byte，则别名为Byte",
          "B. 映射类型为Date，则别名为Date",
          "C. 映射类型为byte，则别名为Byte",
          "D. 映射类型为String，则别名为string"
        ],
        "answer": "D",
        "raw_answer": "D:映射类型为String，则别名为string;"
      },
      {
        "id": 101,
        "question": "下面关于MyBatis提供的用于解决JDBC编程劣势的方案，说法错误的是（）。",
        "type": "multiple_choice",
        "options": [
          "A. 在SqlMapConfig.xml中配置数据链接池，使用连接池管理数据库链接",
          "B. MyBatis自动将Java对象映射至SQL语句，通过Statement中的parameterType定义输入参数的类型",
          "C. MyBatis将SQL语句配置在MyBatis的映射文件中，未与Java代码的分离",
          "D. MyBatis自动将SQL执行结果映射至Java对象，通过Statement中的resultType定义输出结果的类型"
        ],
        "answer": "C",
        "raw_answer": "C:MyBatis将SQL语句配置在MyBatis的映射文件中，未与Java代码的分离;"
      }
    ]
  }
];