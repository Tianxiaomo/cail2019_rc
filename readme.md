![image](img/img_title.png)

# 中国法研杯 CAIL 2019

## 阅读理解赛道
http://cail.cipsc.org.cn/instruction.html
### 1.任务介绍

裁判文书中包含了丰富的案件信息，比如时间、地点、人物关系等等，通过机器智能化地阅读理解裁判文书，可以更快速、便捷地辅助法官、律师以及普通大众获取所需信息。
本任务是首次基于中文裁判文书的阅读理解比赛，属于篇章片段抽取型阅读理解比赛（Span-Extraction Machine Reading Comprehension）。
为了增加问题的多样性，参考英文阅读理解比赛SQuAD和CoQA，本比赛增加了拒答以及是否类（YES/NO）问题。

### 2.数据介绍

本任务技术评测使用的数据集由科大讯飞提供，数据主要来源于裁判文书网公开的裁判文书，其中包含刑事和民事一审裁判文书。
训练集约包含4万个问题，开发集和测试集各约5000个问题。
对于开发集和测试集，每个问题包含3个人工标注参考答案。
鉴于民事和刑事裁判文书在事实描述部分差异性较大，相应的问题类型也不尽相同，为了能同时兼顾这两种裁判文书，从而覆盖大多数裁判文书，本次比赛会设置民事和刑事两类测试集。
数据集详细介绍可以参见https://github.com/china-ai-law-challenge/CAIL2019/tree/master/阅读理解

### 3.评价方式

本任务采用与CoQA比赛一致的宏平均（macro-average F1）进行评估。
对于每个问题，需要与N个标准回答计算得到N个F1，并取最大值作为其F1值。
然而在评估人类表现（Human Performance）的时候，每个标准回答需要与N-1个其它标准回答计算F1值。
为了更公平地对比指标，需要把N个标准回答按照N-1一组的方式分成N组，最终每个问题的F1值为这N组F1的平均值。
整个数据集的F1值为所有数据F1的平均值。
更详细的评价方法详见https://github.com/china-ai-law-challenge/CAIL2019/tree/master/阅读理解

------------------------
开始用keras随便写了个模型,分数惨不忍睹,后来改模型,发现trian着突然结果Nan.后来改成pytorch了,想看看哪里导致的Nan,发现没有Nan了,但是分数上不去,不开心了,看了下BiDAF, R-net等,最后发现还是
Bert厉害,所以无脑上Bert,并没有针对YES和NO的回答做出调整,但是现在过了baseline了,看了看哈工大的AoA,好像可以.

## 相关参考

1.https://rajpurkar.github.io/SQuAD-explorer/