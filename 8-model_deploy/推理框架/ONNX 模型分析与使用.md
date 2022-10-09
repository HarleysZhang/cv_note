> 本文大部分内容为对 ONNX 官方资料的总结和翻译，部分知识点参考网上质量高的博客。
## 一，ONNX 概述
深度学习算法大多通过计算数据流图来完成神经网络的深度学习过程。 一些框架（例如CNTK，Caffe2，Theano和TensorFlow）使用**静态图形**，而其他框架（例如 PyTorch 和 Chainer）使用**动态图形**。 但是这些框架都提供了接口，使开发人员可以轻松构建计算图和运行时，以优化的方式处理图。 这些图用作中间表示（IR），捕获开发人员源代码的特定意图，有助于优化和转换在特定设备（CPU，GPU，FPGA等）上运行。

**ONNX 的本质只是一套开放的 `ML` 模型标准，模型文件存储的只是网络的拓扑结构和权重（其实每个深度学习框架最后保存的模型都是类似的），脱离开框架是没办法对模型直接进行 `inference`的**。
### 1.1，为什么使用通用 IR
现在很多的深度学习框架提供的功能都是类似的，但是在 API、计算图和 runtime 方面却是独立的，这就给 AI 开发者在不同平台部署不同模型带来了很多困难和挑战，ONNX 的目的在于提供一个跨框架的模型中间表达框架，用于模型转换和部署。ONNX 提供的计算图是通用的，格式也是开源的。
## 二，ONNX 规范
> Open Neural Network Exchange Intermediate Representation (ONNX IR) Specification.

`ONNX` 结构的定义文件 `.proto` 和 `.prpto3` 可以在 [onnx folder](https://github.com/onnx/onnx/tree/master/onnx) 目录下找到，文件遵循的是谷歌 `Protobuf` 协议。ONNX 是一个开放式规范，由以下组件组成：
+ 可扩展计算图模型的定义
+ 标准数据类型的定义
+ 内置运算符的定义

`IR6` 版本的 ONNX 只能用于推理（inference），从 `IR7` 开始 ONNX 支持训练（training）。`onnx.proto` 主要的对象如下：
+ ModelProto
+ GraphProto
+ NodeProto
+ AttributeProto
+ ValueInfoProto
+ TensorProto

他们之间的关系：ONNX 模型 `load` 之后，得到的是一个 `ModelProto`，它包含了一些版本信息，生产者信息和一个非常重要的 `GraphProto`；在 `GraphProto` 中包含了四个关键的 `repeated` 数组，分别是`node` (NodeProto 类型)，`input`(ValueInfoProto 类型)，`output`(ValueInfoProto 类型)和 `initializer` (TensorProto 类型)，其中 `node` 中存放着模型中的所有计算节点，input 中存放着模型所有的输入节点，output 存放着模型所有的输出节点，`initializer` 存放着模型所有的权重；节点与节点之间的拓扑定义可以通过 input 和output 这两个 `string` 数组的指向关系得到，这样利用上述信息我们可以快速构建出一个深度学习模型的拓扑图。最后每个计算节点当中还包含了一个 `AttributeProto` 数组，用于描述该节点的属性，例如 `Conv` 层的属性包含 `group`，`pads` 和`strides` 等等，具体每个计算节点的属性、输入和输出可以参考这个 [Operators.md](https://github.com/onnx/onnx/blob/master/docs/Operators.md) 文档。

需要注意的是，上面所说的 `GraphProto` 中的 `input` 输入数组不仅仅包含我们一般理解中的图片输入的那个节点，还包含了模型当中所有权重。举例，`Conv` 层中的 `W` 权重实体是保存在 `initializer` 当中的，那么相应的会有一个同名的输入在 `input` 当中，其背后的逻辑应该是把权重也看作是模型的输入，并通过 `initializer` 中的权重实体来对这个输入做初始化(也就是把值填充进来)
### 2.1，Model
模型结构的主要目的是将元数据( `meta data`)与图形(`graph`)相关联，图形包含所有可执行元素。 首先，读取模型文件时使用元数据，为实现提供所需的信息，以确定它是否能够：执行模型，生成日志消息，错误报告等功能。此外元数据对工具很有用，例如IDE和模型库，它需要它来告知用户给定模型的目的和特征。

每个 model 有以下组件：
|Name|Type|Description|
|---|---|---|
|ir_version|int64|The ONNX version assumed by the model.|
|opset_import|OperatorSetId|A collection of operator set identifiers made available to the model. An implementation must support all operators in the set or reject the model.|
|producer_name|string|The name of the tool used to generate the model.|
|producer_version|string|The version of the generating tool.|
|domain|string|A reverse-DNS name to indicate the model namespace or domain, for example, 'org.onnx'|
|model_version|int64|The version of the model itself, encoded in an integer.|
|doc_string|string|Human-readable documentation for this model. Markdown is allowed.|
|graph|Graph|The parameterized graph that is evaluated to execute the model.|
|metadata_props|map<string,string>|Named metadata values; keys should be distinct.|
|training_info|TrainingInfoProto[]|An optional extension that contains information for training.|
### 2.2，Operators Sets
每个模型必须明确命名它依赖于其功能的运算符集。 操作员集定义可用的操作符，其版本和状态。 每个模型按其域定义导入的运算符集。 所有模型都隐式导入默认的 ONNX 运算符集。

运算符集(`Operators Sets`)对象的属性如下：
|Name|Type|Description|
|---|---|---|
|magic|string|T ‘ONNXOPSET’|
|ir_version|int32|The ONNX version corresponding to the operators.|
|ir_version_prerelease|string|The prerelease component of the SemVer of the IR.
|ir_build_metadata|string|The build metadata of this version of the operator set.|
|domain|string|The domain of the operator set. Must be unique among all sets.|
|opset_version|int64|The version of the operator set.|
|doc_string|string|Human-readable documentation for this operator set. Markdown is allowed.|
|operator|Operator[]|The operators contained in this operator set.|

### 2.3，ONNX Operator
图( `graph`)中使用的每个运算符必须由模型(`model`)导入的一个运算符集明确声明。

运算符（`Operator`）对象定义的属性如下：
|Name|Type|Description|
|---|---|---|
|op_type|string|The name of the operator, as used in graph nodes. MUST be unique within the operator set’s domain.|
|since_version|int64|The version of the operator set when this operator was introduced.|
|status|OperatorStatus|One of ‘EXPERIMENTAL’ or ‘STABLE.’|
|doc_string|string|A human-readable documentation string for this operator. Markdown is allowed.|

### 2.4，ONNX Graph

序列化图由一组元数据字段(`metadata`)，模型参数列表(`a list of model parameters`,)和计算节点列表组成(a list of computation nodes)。每个计算数据流图被构造为拓扑排序的节点列表，这些节点形成图形，其必须没有周期。 每个节点代表对运营商的呼叫。 每个节点具有零个或多个输入以及一个或多个输出。

图表(Graph)对象具有以下属性：

|Name|Type|Description|
|---|---|---|
|name|string|模型计算图的名称|
|node|Node[]|节点列表，基于输入/输出数据依存关系形成部分排序的计算图，拓扑顺序排列。|
|initializer|Tensor[]|命名张量值的列表。 当 ` initializer` 与计算图 `graph`输入名称相同，输入指定一个默认值，否则指定一个常量值。|
|doc_string|string|用于阅读模型的文档|
|input|ValueInfo[]|计算图 `graph` 的输入参数，在 `‘initializer.’` 中可能能找到默认的初始化值。|
|output|ValueInfo[]|计算图 `graph` 的输出参数。|
|value_info|ValueInfo[]|用于存储除输入、输出值之外的类型和形状信息。|

### 2.5，ValueInfo
`ValueInfo` 对象属性如下：

|Name|Type|Description|
|---|---|---|
|name|string|The name of the value/parameter.|
|type|Type|The type of the value **including shape information**.|
|doc_string|string|Human-readable documentation for this value. Markdown is allowed.|
### 2.6，Standard data types
ONNX 标准有两个版本，主要区别在于支持的数据类型和算子不同。计算图 `graphs`、节点 `nodes`和计算图的 `initializers` 支持的数据类型如下。原始数字，字符串和布尔类型必须用作张量的元素。
#### 2.6.1，Tensor Element Types
|Group|Types|Description|
|---|---|---|
|Floating Point Types|float16, float32, float64|浮点数遵循IEEE 754-2008标准。|
|Signed Integer Types|int8, int16, int32, int64|支持 `8-64` 位宽的有符号整数。|
|Unsigned Integer Types|uint8, uint16|支持 `8` 或 `16` 位的无符号整数。|
|Complex Types|complex64, complex128|具有 `32` 位或 `64` 位实部和虚部的复数。|
|Other|string|字符串代表的文本数据。 所有字符串均使用UTF-8编码。|
|Other|bool|布尔值类型，表示的数据只有两个值，通常为 `true` 和 `false`。|
#### 2.6.2，Input / Output Data Types
以下类型用于定义计算图和节点输入和输出的类型。

|Variant | Type | Description |
|---|---|---|
|ONNX|dense tensors|张量是向量和矩阵的一般化|
|ONNX|sequence| `sequence` (序列)是有序的稠密元素集合。|
|ONNX|map|映射是关联表，由键类型和值类型定义。|

**`ONNX` 现阶段没有定义稀疏张量类型**。
## 三，ONNX版本控制

## 四，主要算子概述

## 五，Python API 使用

### 5.1，加载模型

1，**Loading an ONNX model**

```python
import onnx
# onnx_model is an in-mempry ModelProto
onnx_model = onnx.load('path/to/the/model.onnx') # 加载 onnx 模型
```

2，**Loading an ONNX Model with External Data**

+ 【默认加载模型方式】如果外部数据(`external data`)和模型文件在同一个目录下，仅使用 `onnx.load()` 即可加载模型，方法见上小节。
+ 如果外部数据(`external data`)和模型文件不在同一个目录下，在使用 `onnx_load()` 函数后还需使用 `load_external_data_for_model()` 函数指定外部数据路径。

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

onnx_model = onnx.load('path/to/the/model.onnx', load_external_data=False)
load_external_data_for_model(onnx_model, 'data/directory/path/')
# Then the onnx_model has loaded the external data from the specific directory
```

**3，Converting an ONNX Model to External Data**

```python
from onnx.external_data_helper import convert_model_to_external_data

# onnx_model is an in-memory ModelProto
onnx_model = ...
convert_model_to_external_data(onnx_model, all_tensors_to_one_file=True, location='filename', size_threshold=1024, convert_attribute=False)
# Then the onnx_model has converted raw data as external data
# Must be followed by save
```

### 5.2，保存模型

**1，Saving an ONNX Model**
```python
import onnx

# onnx_model is an in-memory ModelProto
onnx_model = ...

# Save the ONNX model
onnx.save(onnx_model, 'path/to/the/model.onnx')
```
2，**Converting and Saving an ONNX Model to External Data**
```python
import onnx

# onnx_model is an in-memory ModelProto
onnx_model = ...
onnx.save_model(onnx_model, 'path/to/save/the/model.onnx', save_as_external_data=True, all_tensors_to_one_file=True, location='filename', size_threshold=1024, convert_attribute=False)
# Then the onnx_model has converted raw data as external data and saved to specific directory
```
### 5.3，Manipulating TensorProto and Numpy Array
```python
import numpy
import onnx
from onnx import numpy_helper

# Preprocessing: create a Numpy array
numpy_array = numpy.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
print('Original Numpy array:\n{}\n'.format(numpy_array))

# Convert the Numpy array to a TensorProto
tensor = numpy_helper.from_array(numpy_array)
print('TensorProto:\n{}'.format(tensor))

# Convert the TensorProto to a Numpy array
new_array = numpy_helper.to_array(tensor)
print('After round trip, Numpy array:\n{}\n'.format(new_array))

# Save the TensorProto
with open('tensor.pb', 'wb') as f:
    f.write(tensor.SerializeToString())

# Load a TensorProto
new_tensor = onnx.TensorProto()
with open('tensor.pb', 'rb') as f:
    new_tensor.ParseFromString(f.read())
print('After saving and loading, new TensorProto:\n{}'.format(new_tensor))
```
### 5.4，创建ONNX模型
可以通过 `helper` 模块提供的函数 `helper.make_graph` 完成创建 ONNX 格式的模型。创建 `graph` 之前，需要先创建相应的 `NodeProto(node)`，参照文档设定节点的属性，指定该节点的输入与输出，如果该节点带有权重那还需要创建相应的`ValueInfoProto` 和 `TensorProto` 分别放入 `graph` 中的 `input` 和 `initializer` 中，以上步骤缺一不可。
```python
import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/master/onnx/onnx.proto

# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [3, 2])
pads = helper.make_tensor_value_info('pads', TensorProto.FLOAT, [1, 4])

value = helper.make_tensor_value_info('value', AttributeProto.FLOAT, [1])

# Create one output (ValueInfoProto)
Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [3, 4])

# Create a node (NodeProto) - This is based on Pad-11
node_def = helper.make_node(
    'Pad',                  # name
    ['X', 'pads', 'value'], # inputs
    ['Y'],                  # outputs
    mode='constant',        # attributes
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],        # nodes
    'test-model',      # name
    [X, pads, value],  # inputs
    [Y],               # outputs
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name='onnx-example')

print('The model is:\n{}'.format(model_def))
onnx.checker.check_model(model_def)
print('The model is checked!')
```
### 5.5，检查模型
在完成 `ONNX` 模型加载或者创建后，有必要对模型进行检查，使用 `onnx.check.check_model()` 函数。
```python
import onnx

# Preprocessing: load the ONNX model
model_path = 'path/to/the/model.onnx'
onnx_model = onnx.load(model_path)

print('The model is:\n{}'.format(onnx_model))

# Check the model
try:
    onnx.checker.check_model(onnx_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')
```
### 5.6，实用功能函数
函数 `extract_model()` 可以从 `ONNX` 模型中提取子模型，子模型由输入和输出张量的名称定义。这个功能方便我们 `debug` 原模型和转换后的 `ONNX` 模型输出结果是否一致(误差小于某个阈值)，不再需要我们手动去修改 `ONNX` 模型。
```python
import onnx

input_path = 'path/to/the/original/model.onnx'
output_path = 'path/to/save/the/extracted/model.onnx'
input_names = ['input_0', 'input_1', 'input_2']
output_names = ['output_0', 'output_1']

onnx.utils.extract_model(input_path, output_path, input_names, output_names)
```
### 5.7，工具
函数 `update_inputs_outputs_dims()` 可以将模型输入和输出的维度更新为参数中指定的值，可以使用 `dim_param` 提供静态和动态尺寸大小。
```python
import onnx
from onnx.tools import update_model_dims

model = onnx.load('path/to/the/model.onnx')
# Here both 'seq', 'batch' and -1 are dynamic using dim_param.
variable_length_model = update_model_dims.update_inputs_outputs_dims(model, {'input_name': ['seq', 'batch', 3, -1]}, {'output_name': ['seq', 'batch', 1, -1]})
# need to check model after the input/output sizes are updated
onnx.checker.check_model(variable_length_model )
```
## 参考资料
1. [ONNX--跨框架的模型中间表达框架](https://zhuanlan.zhihu.com/p/41255090)
2. [深度学习模型转换与部署那些事(含ONNX格式详细分析)](https://bindog.github.io/blog/2020/03/13/deep-learning-model-convert-and-depoly/)
3. [onnx](https://github.com/onnx/tutorials)