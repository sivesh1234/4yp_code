уд;
Ф§
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeѕ
Й
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ѕ
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeѕ"serve*2.0.02v2.0.0-rc2-26-g64c3d382ca8ё╦9
t
dense/kernelVarHandleOp*
shape
:@*
shared_namedense/kernel*
dtype0*
_output_shapes
: 
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
dtype0*
_output_shapes

:@
l

dense/biasVarHandleOp*
dtype0*
_output_shapes
: *
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
dtype0*
_output_shapes
:
l
RMSprop/iterVarHandleOp*
shared_nameRMSprop/iter*
dtype0	*
_output_shapes
: *
shape: 
e
 RMSprop/iter/Read/ReadVariableOpReadVariableOpRMSprop/iter*
dtype0	*
_output_shapes
: 
n
RMSprop/decayVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameRMSprop/decay
g
!RMSprop/decay/Read/ReadVariableOpReadVariableOpRMSprop/decay*
dtype0*
_output_shapes
: 
~
RMSprop/learning_rateVarHandleOp*&
shared_nameRMSprop/learning_rate*
dtype0*
_output_shapes
: *
shape: 
w
)RMSprop/learning_rate/Read/ReadVariableOpReadVariableOpRMSprop/learning_rate*
dtype0*
_output_shapes
: 
t
RMSprop/momentumVarHandleOp*
dtype0*
_output_shapes
: *
shape: *!
shared_nameRMSprop/momentum
m
$RMSprop/momentum/Read/ReadVariableOpReadVariableOpRMSprop/momentum*
dtype0*
_output_shapes
: 
j
RMSprop/rhoVarHandleOp*
dtype0*
_output_shapes
: *
shape: *
shared_nameRMSprop/rho
c
RMSprop/rho/Read/ReadVariableOpReadVariableOpRMSprop/rho*
dtype0*
_output_shapes
: 
s
lstm/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	ђ*
shared_namelstm/kernel
l
lstm/kernel/Read/ReadVariableOpReadVariableOplstm/kernel*
dtype0*
_output_shapes
:	ђ
Є
lstm/recurrent_kernelVarHandleOp*
shape:	@ђ*&
shared_namelstm/recurrent_kernel*
dtype0*
_output_shapes
: 
ђ
)lstm/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm/recurrent_kernel*
dtype0*
_output_shapes
:	@ђ
k
	lstm/biasVarHandleOp*
shared_name	lstm/bias*
dtype0*
_output_shapes
: *
shape:ђ
d
lstm/bias/Read/ReadVariableOpReadVariableOp	lstm/bias*
dtype0*
_output_shapes	
:ђ
w
lstm_1/kernelVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@ђ*
shared_namelstm_1/kernel
p
!lstm_1/kernel/Read/ReadVariableOpReadVariableOplstm_1/kernel*
dtype0*
_output_shapes
:	@ђ
І
lstm_1/recurrent_kernelVarHandleOp*(
shared_namelstm_1/recurrent_kernel*
dtype0*
_output_shapes
: *
shape:	@ђ
ё
+lstm_1/recurrent_kernel/Read/ReadVariableOpReadVariableOplstm_1/recurrent_kernel*
dtype0*
_output_shapes
:	@ђ
o
lstm_1/biasVarHandleOp*
shared_namelstm_1/bias*
dtype0*
_output_shapes
: *
shape:ђ
h
lstm_1/bias/Read/ReadVariableOpReadVariableOplstm_1/bias*
dtype0*
_output_shapes	
:ђ
ї
RMSprop/dense/kernel/rmsVarHandleOp*
shape
:@*)
shared_nameRMSprop/dense/kernel/rms*
dtype0*
_output_shapes
: 
Ё
,RMSprop/dense/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/kernel/rms*
dtype0*
_output_shapes

:@
ё
RMSprop/dense/bias/rmsVarHandleOp*'
shared_nameRMSprop/dense/bias/rms*
dtype0*
_output_shapes
: *
shape:
}
*RMSprop/dense/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/dense/bias/rms*
dtype0*
_output_shapes
:
І
RMSprop/lstm/kernel/rmsVarHandleOp*(
shared_nameRMSprop/lstm/kernel/rms*
dtype0*
_output_shapes
: *
shape:	ђ
ё
+RMSprop/lstm/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/lstm/kernel/rms*
dtype0*
_output_shapes
:	ђ
Ъ
!RMSprop/lstm/recurrent_kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@ђ*2
shared_name#!RMSprop/lstm/recurrent_kernel/rms
ў
5RMSprop/lstm/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp!RMSprop/lstm/recurrent_kernel/rms*
dtype0*
_output_shapes
:	@ђ
Ѓ
RMSprop/lstm/bias/rmsVarHandleOp*
shape:ђ*&
shared_nameRMSprop/lstm/bias/rms*
dtype0*
_output_shapes
: 
|
)RMSprop/lstm/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/lstm/bias/rms*
dtype0*
_output_shapes	
:ђ
Ј
RMSprop/lstm_1/kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@ђ**
shared_nameRMSprop/lstm_1/kernel/rms
ѕ
-RMSprop/lstm_1/kernel/rms/Read/ReadVariableOpReadVariableOpRMSprop/lstm_1/kernel/rms*
dtype0*
_output_shapes
:	@ђ
Б
#RMSprop/lstm_1/recurrent_kernel/rmsVarHandleOp*
dtype0*
_output_shapes
: *
shape:	@ђ*4
shared_name%#RMSprop/lstm_1/recurrent_kernel/rms
ю
7RMSprop/lstm_1/recurrent_kernel/rms/Read/ReadVariableOpReadVariableOp#RMSprop/lstm_1/recurrent_kernel/rms*
dtype0*
_output_shapes
:	@ђ
Є
RMSprop/lstm_1/bias/rmsVarHandleOp*
shape:ђ*(
shared_nameRMSprop/lstm_1/bias/rms*
dtype0*
_output_shapes
: 
ђ
+RMSprop/lstm_1/bias/rms/Read/ReadVariableOpReadVariableOpRMSprop/lstm_1/bias/rms*
dtype0*
_output_shapes	
:ђ

NoOpNoOp
Ы$
ConstConst"/device:CPU:0*Г$
valueБ$Bа$ BЎ$
з
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer

signatures
	variables
regularization_losses
		keras_api

trainable_variables
R
	variables
regularization_losses
	keras_api
trainable_variables
l
cell

state_spec
	variables
regularization_losses
	keras_api
trainable_variables
l
cell

state_spec
	variables
regularization_losses
	keras_api
trainable_variables
h

kernel
bias
	variables
regularization_losses
	keras_api
 trainable_variables
Ќ
!iter
	"decay
#learning_rate
$momentum
%rho	rmsP	rmsQ	&rmsR	'rmsS	(rmsT	)rmsU	*rmsV	+rmsW
 
8
&0
'1
(2
)3
*4
+5
6
7
 
џ

,layers

trainable_variables
	variables
-layer_regularization_losses
.metrics
regularization_losses
/non_trainable_variables
8
&0
'1
(2
)3
*4
+5
6
7
 
 
џ

0layers
trainable_variables
	variables
1layer_regularization_losses
2metrics
regularization_losses
3non_trainable_variables
 
~

&kernel
'recurrent_kernel
(bias
4	variables
5regularization_losses
6	keras_api
7trainable_variables
 

&0
'1
(2
 
џ

8layers
trainable_variables
	variables
9layer_regularization_losses
:metrics
regularization_losses
;non_trainable_variables

&0
'1
(2
~

)kernel
*recurrent_kernel
+bias
<	variables
=regularization_losses
>	keras_api
?trainable_variables
 

)0
*1
+2
 
џ

@layers
trainable_variables
	variables
Alayer_regularization_losses
Bmetrics
regularization_losses
Cnon_trainable_variables

)0
*1
+2
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
џ

Dlayers
 trainable_variables
	variables
Elayer_regularization_losses
Fmetrics
regularization_losses
Gnon_trainable_variables

0
1
KI
VARIABLE_VALUERMSprop/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
MK
VARIABLE_VALUERMSprop/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
][
VARIABLE_VALUERMSprop/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUERMSprop/momentum-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUERMSprop/rho(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUElstm/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUElstm/recurrent_kernel&variables/1/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE	lstm/bias&variables/2/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUElstm_1/kernel&variables/3/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUElstm_1/recurrent_kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUElstm_1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
 
 
 
 
 

&0
'1
(2
 
џ

Hlayers
7trainable_variables
4	variables
Ilayer_regularization_losses
Jmetrics
5regularization_losses
Knon_trainable_variables

&0
'1
(2

0
 
 
 

)0
*1
+2
 
џ

Llayers
?trainable_variables
<	variables
Mlayer_regularization_losses
Nmetrics
=regularization_losses
Onon_trainable_variables

)0
*1
+2

0
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
Ѓђ
VARIABLE_VALUERMSprop/dense/kernel/rmsTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
~|
VARIABLE_VALUERMSprop/dense/bias/rmsRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUERMSprop/lstm/kernel/rmsDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE!RMSprop/lstm/recurrent_kernel/rmsDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
om
VARIABLE_VALUERMSprop/lstm/bias/rmsDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUERMSprop/lstm_1/kernel/rmsDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUE#RMSprop/lstm_1/recurrent_kernel/rmsDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUERMSprop/lstm_1/bias/rmsDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
: 
Ё
serving_default_lstm_inputPlaceholder* 
shape:         Z*
dtype0*+
_output_shapes
:         Z
б
StatefulPartitionedCallStatefulPartitionedCallserving_default_lstm_inputlstm/kernellstm/recurrent_kernel	lstm/biaslstm_1/kernellstm_1/recurrent_kernellstm_1/biasdense/kernel
dense/bias*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         *
Tin
2	*,
_gradient_op_typePartitionedCall-15410*,
f'R%
#__inference_signature_wrapper_11313
O
saver_filenamePlaceholder*
shape: *
dtype0*
_output_shapes
: 
В
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp RMSprop/iter/Read/ReadVariableOp!RMSprop/decay/Read/ReadVariableOp)RMSprop/learning_rate/Read/ReadVariableOp$RMSprop/momentum/Read/ReadVariableOpRMSprop/rho/Read/ReadVariableOplstm/kernel/Read/ReadVariableOp)lstm/recurrent_kernel/Read/ReadVariableOplstm/bias/Read/ReadVariableOp!lstm_1/kernel/Read/ReadVariableOp+lstm_1/recurrent_kernel/Read/ReadVariableOplstm_1/bias/Read/ReadVariableOp,RMSprop/dense/kernel/rms/Read/ReadVariableOp*RMSprop/dense/bias/rms/Read/ReadVariableOp+RMSprop/lstm/kernel/rms/Read/ReadVariableOp5RMSprop/lstm/recurrent_kernel/rms/Read/ReadVariableOp)RMSprop/lstm/bias/rms/Read/ReadVariableOp-RMSprop/lstm_1/kernel/rms/Read/ReadVariableOp7RMSprop/lstm_1/recurrent_kernel/rms/Read/ReadVariableOp+RMSprop/lstm_1/bias/rms/Read/ReadVariableOpConst*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *"
Tin
2	*,
_gradient_op_typePartitionedCall-15453*'
f"R 
__inference__traced_save_15452
├
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasRMSprop/iterRMSprop/decayRMSprop/learning_rateRMSprop/momentumRMSprop/rholstm/kernellstm/recurrent_kernel	lstm/biaslstm_1/kernellstm_1/recurrent_kernellstm_1/biasRMSprop/dense/kernel/rmsRMSprop/dense/bias/rmsRMSprop/lstm/kernel/rms!RMSprop/lstm/recurrent_kernel/rmsRMSprop/lstm/bias/rmsRMSprop/lstm_1/kernel/rms#RMSprop/lstm_1/recurrent_kernel/rmsRMSprop/lstm_1/bias/rms**
f%R#
!__inference__traced_restore_15528*
Tout
2**
config_proto

GPU 

CPU2J 8*
_output_shapes
: *!
Tin
2*,
_gradient_op_typePartitionedCall-15529Гж8
Х
█
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_15336

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђp
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0U
mulMulSigmoid_1:y:0states_1*'
_output_shapes
:         @*
T0N
ReluRelusplit:output:2*'
_output_shapes
:         @*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         @*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:         @:         @:         @:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
Г
п
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9289

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:         @:         @:         @:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
а>
╝
*__inference_cudnn_lstm_with_fallback_12931

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB: C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
dtype0*
valueB: E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_fa5d5f1a-3419-4659-95e4-7ff9aa932079*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
пP
б
A__inference_lstm_1_layer_call_and_return_conditional_losses_10998

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
dtype0*
_output_shapes
: *
value
B :У_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Z         @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђ|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Д
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         @*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_10897*L
_output_shapes:
8: : : : :         @:         @: : : : : *K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_10896K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         @*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
         a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
Index0*
T0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *    │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*6
_input_shapes%
#:         Z@:::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs: : : 
╬+
ь
while_body_8810
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         @*
T0K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : : : :	 :
 :  : 
ЇQ
ц
A__inference_lstm_1_layer_call_and_return_conditional_losses_15237
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
_output_shapes
: *
value	B :@*
dtype0s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
_output_shapes
:*
valueB:*
dtype0█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"    @   *
dtype0═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
_output_shapes
:*
valueB:*
dtype0a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђ|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
_output_shapes
: *
value	B : *
dtype0Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_15138*L
_output_shapes:
8: : : : :         @:         @: : : : : *K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_15137*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         @*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  @*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:( $
"
_user_specified_name
inputs/0: : : 
а>
╝
*__inference_cudnn_lstm_with_fallback_10151

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
dtype0*
valueB: E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
dtype0*
valueB:к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*
_input_shapes *=
api_implements+)lstm_4e96b1c4-3fa5-495d-9170-91a62470ff34*
api_preferred_deviceGPU2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
▄
њ
while_cond_14970
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
м┼
¤
;__inference___backward_cudnn_lstm_with_fallback_13403_13585
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*4
_output_shapes"
 :                  @*
Index0*
T0*
shrink_axis_maskю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:└
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
Index0*
T0Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
T0*
Index0ѓ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:И
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  :         @:         @:ђїќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
_output_shapes
:*
T0╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:         @*
T0^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
_output_shapes
:*
valueB:ђ*
dtype0j
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
valueB:ђ *
dtype0i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
_output_shapes
:*
valueB:@*
dtype0j
 gradients/concat_1_grad/Shape_12Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
valueB:@*
dtype0j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ђ №
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:@*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_1_grad/ShapeConst*
_output_shapes
:*
valueB"@      *
dtype0Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
_output_shapes

:@@*
T0h
gradients/Reshape_8_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:@*
T0h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
_output_shapes
:*
valueB:@*
dtype0д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_15_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes
:	@ђ*
T0\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђд
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  Ъ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*'
_output_shapes
:         @*
T0Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
_output_shapes	
:ђ*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*║
_input_shapesе
Ц:         @:                  @:         @:         @: :                  @:::::         @::::         @:::::                  :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_62400428-7c83-468c-b0ce-181c88f5395f*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_135842T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : : : : : : : : : :  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : 
дH
ћ
__inference_standard_lstm_8431

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*4
_output_shapes"
 :                  *
T0B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
_output_shapes
:*
valueB:*
dtype0_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:         *
T0*
Index0*
shrink_axis_maskІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
_output_shapes
: *
value	B :*
dtype0Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*'
_output_shapes
:         @*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:         @*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Є
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_8327*
_num_original_outputs*
bodyR
while_body_8328*L
_output_shapes:
8: : : : :         @:         @: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         @*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0╣

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*4
_output_shapes"
 :                  @*
T0Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*e
_input_shapesT
R:                  :         @:         @:::*=
api_implements+)lstm_f89f659c-75ea-4b17-99d9-8c9723ab031f*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
└B
ь
@__inference_lstm_1_layer_call_and_return_conditional_losses_9851

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCallбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
_output_shapes
: *
value	B :@*
dtype0w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*
_output_shapes
:*!
valueB"          *
dtype0v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*
element_dtype0*
_output_shapes
: *

shape_type0є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
valueB"    @   *
dtype0═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0Ж
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9324*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         @:         @:         @*+
_gradient_op_typePartitionedCall-9347n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
_output_shapes
: *
valueB :
         *
dtype0T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: ф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_9770*
_num_original_outputs*
bodyR
while_body_9771*L
_output_shapes:
8: : : : :         @:         @: : : : : K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
_output_shapes
:*
valueB"    @   *
dtype0о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  @*
T0[
runtimeConst"/device:CPU:0*
_output_shapes
: *
valueB
 *    *
dtype0ѓ
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile: : :& "
 
_user_specified_nameinputs: 
лP
Ї
(__forward_cudnn_lstm_with_fallback_12429

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
dtype0*
value	B : H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
dtype0*
valueB: E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"
	transposetranspose_0:y:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0*
_input_shapes *=
api_implements+)lstm_5e66a3d3-3a72-45c8-be76-304c5e537995*
api_preferred_deviceGPU*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_12248_124302,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
¤┼
═
9__inference___backward_cudnn_lstm_with_fallback_9025_9207
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:         @*
T0m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:         @*
T0O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
_output_shapes
:*
T0й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*4
_output_shapes"
 :                  @*
T0*
Index0*
shrink_axis_maskю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:└
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
_output_shapes
:*
T0─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
Index0*
T0Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
T0*
Index0ѓ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  @*
T0c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0И
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*j
_output_shapesX
V:                  :         @:         @:ђї*
T0ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*4
_output_shapes"
 :                  *
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
_output_shapes
:*
T0╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
_output_shapes
:*
valueB:ђ*
dtype0j
gradients/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
_output_shapes
:*
valueB:ђ*
dtype0j
gradients/concat_1_grad/Shape_4Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
_output_shapes
:*
valueB:ђ *
dtype0j
gradients/concat_1_grad/Shape_7Const*
_output_shapes
:*
valueB:ђ *
dtype0i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
_output_shapes
:*
valueB:@*
dtype0j
 gradients/concat_1_grad/Shape_13Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
_output_shapes
:*
valueB:@*
dtype0j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:ђ *
Index0*
T0№
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:@*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:@*
T0h
gradients/Reshape_9_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_10_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:@*
T0ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
_output_shapes
:	ђ*
T0х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	@ђ\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
_output_shapes	
:ђ*
Index0*
T0д
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*4
_output_shapes"
 :                  *
T0Ъ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*B
forward_function_name)'__forward_cudnn_lstm_with_fallback_9206*║
_input_shapesе
Ц:         @:                  @:         @:         @: :                  @:::::         @::::         @:::::                  :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_0408b657-aecf-41bb-983d-774dc27442ad*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :  : 
ц
╦
&__inference_lstm_1_layer_call_fn_14903

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-11179*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_11167*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         @*
Tin
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*6
_input_shapes%
#:         Z@:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
и

Ѓ
*__inference_sequential_layer_call_fn_12616

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*'
_output_shapes
:         *
Tin
2	*,
_gradient_op_typePartitionedCall-11252*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11251*
Tout
2**
config_proto

GPU 

CPU2J 8ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::22
StatefulPartitionedCallStatefulPartitionedCall: : : : : : : :& "
 
_user_specified_nameinputs: 
п
љ
while_cond_13187
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: :	 :
 :  : : : : : : : 
лP
Ї
(__forward_cudnn_lstm_with_fallback_10333

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
dtype0*
valueB"       L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
dtype0*
valueB:
         C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*
dtype0*!
valueB"          O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"
cudnnrnnCudnnRNN:output:0"-
transpose_5_permtranspose_5/perm:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_10152_10334*
_input_shapes *=
api_implements+)lstm_4e96b1c4-3fa5-495d-9170-91a62470ff34*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
Г
п
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9324

inputs

states
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulstatesMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:         @:         @:         @:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_namestates:&"
 
_user_specified_namestates: : : 
▒
═
&__inference_lstm_1_layer_call_fn_15253
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_9851*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         @*+
_gradient_op_typePartitionedCall-9852ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
¤┼
═
9__inference___backward_cudnn_lstm_with_fallback_8543_8725
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  @`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*4
_output_shapes"
 :                  @*
Index0*
T0ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:└
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:         @*
T0*
Index0*
shrink_axis_maskЂ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:         @ѓ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:И
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*j
_output_shapesX
V:                  :         @:         @:ђї*
T0ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*'
_output_shapes
:         @*
T0y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_4Const*
dtype0*
_output_shapes
:*
valueB:ђ j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_11Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:ђ *
Index0*
T0№
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:@*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:@*
T0h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_15_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes
:	@ђ*
T0\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:ђh
gradients/concat_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
_output_shapes	
:ђ*
Index0*
T0д
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  Ъ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
_output_shapes
:	ђ*
T0Њ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*B
forward_function_name)'__forward_cudnn_lstm_with_fallback_8724*║
_input_shapesе
Ц:         @:                  @:         @:         @: :                  @:::::         @::::         @:::::                  :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_f89f659c-75ea-4b17-99d9-8c9723ab031f*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : 
ЄS
┐
!__inference__traced_restore_15528
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias#
assignvariableop_2_rmsprop_iter$
 assignvariableop_3_rmsprop_decay,
(assignvariableop_4_rmsprop_learning_rate'
#assignvariableop_5_rmsprop_momentum"
assignvariableop_6_rmsprop_rho"
assignvariableop_7_lstm_kernel,
(assignvariableop_8_lstm_recurrent_kernel 
assignvariableop_9_lstm_bias%
!assignvariableop_10_lstm_1_kernel/
+assignvariableop_11_lstm_1_recurrent_kernel#
assignvariableop_12_lstm_1_bias0
,assignvariableop_13_rmsprop_dense_kernel_rms.
*assignvariableop_14_rmsprop_dense_bias_rms/
+assignvariableop_15_rmsprop_lstm_kernel_rms9
5assignvariableop_16_rmsprop_lstm_recurrent_kernel_rms-
)assignvariableop_17_rmsprop_lstm_bias_rms1
-assignvariableop_18_rmsprop_lstm_1_kernel_rms;
7assignvariableop_19_rmsprop_lstm_1_recurrent_kernel_rms/
+assignvariableop_20_rmsprop_lstm_1_bias_rms
identity_22ѕбAssignVariableOpбAssignVariableOp_1бAssignVariableOp_10бAssignVariableOp_11бAssignVariableOp_12бAssignVariableOp_13бAssignVariableOp_14бAssignVariableOp_15бAssignVariableOp_16бAssignVariableOp_17бAssignVariableOp_18бAssignVariableOp_19бAssignVariableOp_2бAssignVariableOp_20бAssignVariableOp_3бAssignVariableOp_4бAssignVariableOp_5бAssignVariableOp_6бAssignVariableOp_7бAssignVariableOp_8бAssignVariableOp_9б	RestoreV2бRestoreV2_1 	
RestoreV2/tensor_namesConst"/device:CPU:0*Ц	
valueЏ	Bў	B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:џ
RestoreV2/shape_and_slicesConst"/device:CPU:0*=
value4B2B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:Є
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
2	L
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:y
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0*
dtype0*
_output_shapes
 N

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:}
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0*
dtype0*
_output_shapes
 N

Identity_2IdentityRestoreV2:tensors:2*
_output_shapes
:*
T0	
AssignVariableOp_2AssignVariableOpassignvariableop_2_rmsprop_iterIdentity_2:output:0*
dtype0	*
_output_shapes
 N

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:ђ
AssignVariableOp_3AssignVariableOp assignvariableop_3_rmsprop_decayIdentity_3:output:0*
dtype0*
_output_shapes
 N

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:ѕ
AssignVariableOp_4AssignVariableOp(assignvariableop_4_rmsprop_learning_rateIdentity_4:output:0*
dtype0*
_output_shapes
 N

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:Ѓ
AssignVariableOp_5AssignVariableOp#assignvariableop_5_rmsprop_momentumIdentity_5:output:0*
dtype0*
_output_shapes
 N

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:~
AssignVariableOp_6AssignVariableOpassignvariableop_6_rmsprop_rhoIdentity_6:output:0*
dtype0*
_output_shapes
 N

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:~
AssignVariableOp_7AssignVariableOpassignvariableop_7_lstm_kernelIdentity_7:output:0*
dtype0*
_output_shapes
 N

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:ѕ
AssignVariableOp_8AssignVariableOp(assignvariableop_8_lstm_recurrent_kernelIdentity_8:output:0*
dtype0*
_output_shapes
 N

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:|
AssignVariableOp_9AssignVariableOpassignvariableop_9_lstm_biasIdentity_9:output:0*
dtype0*
_output_shapes
 P
Identity_10IdentityRestoreV2:tensors:10*
_output_shapes
:*
T0Ѓ
AssignVariableOp_10AssignVariableOp!assignvariableop_10_lstm_1_kernelIdentity_10:output:0*
dtype0*
_output_shapes
 P
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:Ї
AssignVariableOp_11AssignVariableOp+assignvariableop_11_lstm_1_recurrent_kernelIdentity_11:output:0*
dtype0*
_output_shapes
 P
Identity_12IdentityRestoreV2:tensors:12*
_output_shapes
:*
T0Ђ
AssignVariableOp_12AssignVariableOpassignvariableop_12_lstm_1_biasIdentity_12:output:0*
dtype0*
_output_shapes
 P
Identity_13IdentityRestoreV2:tensors:13*
_output_shapes
:*
T0ј
AssignVariableOp_13AssignVariableOp,assignvariableop_13_rmsprop_dense_kernel_rmsIdentity_13:output:0*
dtype0*
_output_shapes
 P
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:ї
AssignVariableOp_14AssignVariableOp*assignvariableop_14_rmsprop_dense_bias_rmsIdentity_14:output:0*
dtype0*
_output_shapes
 P
Identity_15IdentityRestoreV2:tensors:15*
_output_shapes
:*
T0Ї
AssignVariableOp_15AssignVariableOp+assignvariableop_15_rmsprop_lstm_kernel_rmsIdentity_15:output:0*
dtype0*
_output_shapes
 P
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:Ќ
AssignVariableOp_16AssignVariableOp5assignvariableop_16_rmsprop_lstm_recurrent_kernel_rmsIdentity_16:output:0*
dtype0*
_output_shapes
 P
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:І
AssignVariableOp_17AssignVariableOp)assignvariableop_17_rmsprop_lstm_bias_rmsIdentity_17:output:0*
dtype0*
_output_shapes
 P
Identity_18IdentityRestoreV2:tensors:18*
_output_shapes
:*
T0Ј
AssignVariableOp_18AssignVariableOp-assignvariableop_18_rmsprop_lstm_1_kernel_rmsIdentity_18:output:0*
dtype0*
_output_shapes
 P
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:Ў
AssignVariableOp_19AssignVariableOp7assignvariableop_19_rmsprop_lstm_1_recurrent_kernel_rmsIdentity_19:output:0*
dtype0*
_output_shapes
 P
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:Ї
AssignVariableOp_20AssignVariableOp+assignvariableop_20_rmsprop_lstm_1_bias_rmsIdentity_20:output:0*
dtype0*
_output_shapes
 ї
RestoreV2_1/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPHt
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*
valueB
B х
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
21
NoOpNoOp"/device:CPU:0*
_output_shapes
 Ю
Identity_21Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: ф
Identity_22IdentityIdentity_21:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: "#
identity_22Identity_22:output:0*i
_input_shapesX
V: :::::::::::::::::::::2*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112
RestoreV2_1RestoreV2_12*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_18AssignVariableOp_182(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV2:+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : : : : : 
ЕH
Ћ
__inference_standard_lstm_12820

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         *
Index0*
T0І
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
parallel_iterations *
condR
while_cond_12716*
_num_original_outputs*
bodyR
while_body_12717*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  @*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @╣

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  @Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*e
_input_shapesT
R:                  :         @:         @:::*=
api_implements+)lstm_fa5d5f1a-3419-4659-95e4-7ff9aa932079*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
пP
б
A__inference_lstm_1_layer_call_and_return_conditional_losses_11167

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:         @*
T0O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         @*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Z         @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         @Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђ|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
parallel_iterations *
condR
while_cond_11065*
_num_original_outputs*
bodyR
while_body_11066*L
_output_shapes:
8: : : : :         @:         @: : : : : *K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0"
identityIdentity:output:0*6
_input_shapes%
#:         Z@:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : 
▓
с
>__inference_lstm_layer_call_and_return_conditional_losses_9209

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @Н
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*o
_output_shapes]
[:         @:                  @:         @:         @: *
Tin

2*+
_gradient_op_typePartitionedCall-8914*'
f"R 
__inference_standard_lstm_8913*
Tout	
2**
config_proto

GPU 

CPU2J 8Ј
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*?
_input_shapes.
,:                  :::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: 
Ё
љ
while_cond_14147
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
─{
Ѓ
E__inference_sequential_layer_call_and_return_conditional_losses_12603

inputs'
#lstm_statefulpartitionedcall_args_3'
#lstm_statefulpartitionedcall_args_4'
#lstm_statefulpartitionedcall_args_5)
%lstm_1_matmul_readvariableop_resource+
'lstm_1_matmul_1_readvariableop_resource*
&lstm_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбlstm/StatefulPartitionedCallбlstm_1/BiasAdd/ReadVariableOpбlstm_1/MatMul/ReadVariableOpбlstm_1/MatMul_1/ReadVariableOpбlstm_1/while@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ж
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskR
lstm/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: T
lstm/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: U
lstm/zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: ѓ
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
N*
_output_shapes
:*
T0U
lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         @T
lstm/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: V
lstm/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
_output_shapes
: *
T0W
lstm/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: є
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @В
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm/zeros:output:0lstm/zeros_1:output:0#lstm_statefulpartitionedcall_args_3#lstm_statefulpartitionedcall_args_4#lstm_statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*
Tin

2*f
_output_shapesT
R:         @:         Z@:         @:         @: *,
_gradient_op_typePartitionedCall-12137*(
f#R!
__inference_standard_lstm_12136*
Tout	
2a
lstm_1/ShapeShape%lstm/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: T
lstm_1/zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: V
lstm_1/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: W
lstm_1/zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: ѕ
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:         @V
lstm_1/zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
_output_shapes
: *
T0X
lstm_1/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: t
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: Y
lstm_1/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: ї
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Є
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @j
lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:џ
lstm_1/transpose	Transpose%lstm/StatefulPartitionedCall:output:1lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:Z         @R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:■
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0m
"lstm_1/TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Х
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ї
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   Р
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: f
lstm_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: h
lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ї
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0▒
lstm_1/MatMul/ReadVariableOpReadVariableOp%lstm_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђЉ
lstm_1/MatMulMatMullstm_1/strided_slice_2:output:0$lstm_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђх
lstm_1/MatMul_1/ReadVariableOpReadVariableOp'lstm_1_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђІ
lstm_1/MatMul_1MatMullstm_1/zeros:output:0&lstm_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz

lstm_1/addAddV2lstm_1/MatMul:product:0lstm_1/MatMul_1:product:0*(
_output_shapes
:         ђ*
T0»
lstm_1/BiasAdd/ReadVariableOpReadVariableOp&lstm_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђЃ
lstm_1/BiasAddBiasAddlstm_1/add:z:0%lstm_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђN
lstm_1/ConstConst*
dtype0*
_output_shapes
: *
value	B :X
lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ╦
lstm_1/splitSplitlstm_1/split/split_dim:output:0lstm_1/BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @b
lstm_1/SigmoidSigmoidlstm_1/split:output:0*
T0*'
_output_shapes
:         @d
lstm_1/Sigmoid_1Sigmoidlstm_1/split:output:1*
T0*'
_output_shapes
:         @r

lstm_1/mulMullstm_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*
T0*'
_output_shapes
:         @\
lstm_1/ReluRelulstm_1/split:output:2*
T0*'
_output_shapes
:         @t
lstm_1/mul_1Mullstm_1/Sigmoid:y:0lstm_1/Relu:activations:0*
T0*'
_output_shapes
:         @i
lstm_1/add_1AddV2lstm_1/mul:z:0lstm_1/mul_1:z:0*
T0*'
_output_shapes
:         @d
lstm_1/Sigmoid_2Sigmoidlstm_1/split:output:3*
T0*'
_output_shapes
:         @Y
lstm_1/Relu_1Relulstm_1/add_1:z:0*
T0*'
_output_shapes
:         @x
lstm_1/mul_2Mullstm_1/Sigmoid_2:y:0lstm_1/Relu_1:activations:0*'
_output_shapes
:         @*
T0u
$lstm_1/TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:║
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: M
lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: a
lstm_1/while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: [
lstm_1/while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : О
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_1_matmul_readvariableop_resource'lstm_1_matmul_1_readvariableop_resource&lstm_1_biasadd_readvariableop_resource^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp*K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *#
condR
lstm_1_while_cond_12495*
_num_original_outputs*#
bodyR
lstm_1_while_body_12496*L
_output_shapes:
8: : : : :         @:         @: : : : : Y
lstm_1/while/IdentityIdentitylstm_1/while:output:0*
T0*
_output_shapes
: [
lstm_1/while/Identity_1Identitylstm_1/while:output:1*
T0*
_output_shapes
: [
lstm_1/while/Identity_2Identitylstm_1/while:output:2*
T0*
_output_shapes
: [
lstm_1/while/Identity_3Identitylstm_1/while:output:3*
T0*
_output_shapes
: l
lstm_1/while/Identity_4Identitylstm_1/while:output:4*'
_output_shapes
:         @*
T0l
lstm_1/while/Identity_5Identitylstm_1/while:output:5*
T0*'
_output_shapes
:         @[
lstm_1/while/Identity_6Identitylstm_1/while:output:6*
T0*
_output_shapes
: [
lstm_1/while/Identity_7Identitylstm_1/while:output:7*
T0*
_output_shapes
: [
lstm_1/while/Identity_8Identitylstm_1/while:output:8*
T0*
_output_shapes
: [
lstm_1/while/Identity_9Identitylstm_1/while:output:9*
_output_shapes
: *
T0]
lstm_1/while/Identity_10Identitylstm_1/while:output:10*
T0*
_output_shapes
: ѕ
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Р
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack lstm_1/while/Identity_3:output:0@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @o
lstm_1/strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ф
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         @l
lstm_1/transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ф
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@b
lstm_1/runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *    «
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ј
dense/MatMulMatMullstm_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         г
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Е
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/StatefulPartitionedCall^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp^lstm_1/while*
T0*'
_output_shapes
:         "
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2<
lstm_1/MatMul/ReadVariableOplstm_1/MatMul/ReadVariableOp2@
lstm_1/MatMul_1/ReadVariableOplstm_1/MatMul_1/ReadVariableOp2>
lstm_1/BiasAdd/ReadVariableOplstm_1/BiasAdd/ReadVariableOp2
lstm_1/whilelstm_1/while: : : :& "
 
_user_specified_nameinputs: : : : : 
а>
╝
*__inference_cudnn_lstm_with_fallback_11603

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
dtype0*
valueB"       L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
dtype0*
valueB"       N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
dtype0*
valueB:к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_74cdc0f2-ebae-400d-825a-4e762059078b*
api_preferred_deviceGPU20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
в+
Ы
while_body_15138
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
ы
ћ
E__inference_sequential_layer_call_and_return_conditional_losses_11251

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallЪ
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tin
2*+
_output_shapes
:         Z@*,
_gradient_op_typePartitionedCall-10337*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_10336*
Tout
2**
config_proto

GPU 

CPU2J 8─
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         @*
Tin
2*,
_gradient_op_typePartitionedCall-11170*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_10998*
Tout
2џ
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11204*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11203*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         *
Tin
2╬
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
═P
ї
'__forward_cudnn_lstm_with_fallback_9206

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
dtype0*
value	B : H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
dtype0*
valueBђ*    ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
dtype0*
valueB:
         E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0o
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*
dtype0*!
valueB"          O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
dtype0*
valueB: E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0*U
backward_function_name;9__inference___backward_cudnn_lstm_with_fallback_9025_9207*
_input_shapes *=
api_implements+)lstm_0408b657-aecf-41bb-983d-774dc27442ad*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
█
Љ
while_cond_9630
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::
 :  : : : : : : : : :	 
▄
њ
while_cond_15137
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::  : : : : : : : : :	 :
 
а>
╝
*__inference_cudnn_lstm_with_fallback_14364

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
dtype0*
valueB"       N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
dtype0*
value	B : ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*
dtype0*!
valueB"          O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
dtype0*
valueB:к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_f39847c2-7a35-4198-9536-241cf3e2865c*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
═P
ї
'__forward_cudnn_lstm_with_fallback_7132

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
dtype0*
valueBђ*    ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
dtype0*
valueB:к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0*
_input_shapes *=
api_implements+)lstm_b2b5e826-16f3-4821-93bd-08aa6fbc38a4*
api_preferred_deviceGPU*U
backward_function_name;9__inference___backward_cudnn_lstm_with_fallback_6951_713326
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ѕ
њ
while_cond_10896
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
м┼
¤
;__inference___backward_cudnn_lstm_with_fallback_12932_13114
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @m
gradients/grad_ys_1Identityplaceholder_1*
T0*4
_output_shapes"
 :                  @`
gradients/grad_ys_2Identityplaceholder_2*'
_output_shapes
:         @*
T0`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:         @*
T0O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
_output_shapes
:*
T0й
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*4
_output_shapes"
 :                  @*
T0*
Index0ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:└
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
T0*
Index0Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         @ѓ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*4
_output_shapes"
 :                  @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:И
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*j
_output_shapesX
V:                  :         @:         @:ђїќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:┘
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*4
_output_shapes"
 :                  u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*'
_output_shapes
:         @*
T0y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ђ №
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:@*
Index0*
T0№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:@*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_5_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_10_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_15_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
_output_shapes	
:ђ*
T0»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	@ђ\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђд
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*4
_output_shapes"
 :                  Ъ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*'
_output_shapes
:         @*
T0А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
_output_shapes
:	@ђ*
T0Ј

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_13113*║
_input_shapesе
Ц:         @:                  @:         @:         @: :                  @:::::         @::::         @:::::                  :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_fa5d5f1a-3419-4659-95e4-7ff9aa932079*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
а>
╝
*__inference_cudnn_lstm_with_fallback_13402

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          @
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
dtype0*
valueB: E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_62400428-7c83-468c-b0ce-181c88f5395f*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel
¤+
Ь
while_body_12031
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         @*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         @*
T0K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:         @*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : : : :	 :
 :  : 
Ё
љ
while_cond_13674
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
_output_shapes
: *
T0]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::
 :  : : : : : : : : :	 
§
ў
E__inference_sequential_layer_call_and_return_conditional_losses_11216

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallБ
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_10336*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         Z@*,
_gradient_op_typePartitionedCall-10337─
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-11170*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_10998*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         @џ
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11204*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11203*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         ╬
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:* &
$
_user_specified_name
lstm_input: : : : : : : : 
в+
Ы
while_body_11066
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         @*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :	 :
 :  : : : 
К
╦
$__inference_lstm_layer_call_fn_12637
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tout
2**
config_proto

GPU 

CPU2J 8*4
_output_shapes"
 :                  @*
Tin
2*+
_gradient_op_typePartitionedCall-8728*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_8727Ј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*?
_input_shapes.
,:                  :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
Э─
¤
;__inference___backward_cudnn_lstm_with_fallback_14365_14547
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*+
_output_shapes
:         Z@*
T0`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:┤
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*+
_output_shapes
:Z         @*
T0*
Index0ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:и
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:Z         @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*+
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskЂ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*+
_output_shapes
:         @*
T0*
Index0*
shrink_axis_maskщ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:Z         @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:»
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*a
_output_shapesO
M:Z         :         @:         @:ђї*
T0ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         Zu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_4Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ђ №
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_3_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
_output_shapes

:@@*
T0h
gradients/Reshape_8_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_10_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_14_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
_output_shapes	
:ђ*
T0»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	@ђ\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:ђh
gradients/concat_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*+
_output_shapes
:         Z*
T0Ъ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*'
_output_shapes
:         @*
T0Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
_output_shapes	
:ђ*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_14546*Ъ
_input_shapesЇ
і:         @:         Z@:         @:         @: :Z         @:::::         @::::         @:::::Z         :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_f39847c2-7a35-4198-9536-241cf3e2865c*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : : : :  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : 
├

Є
*__inference_sequential_layer_call_fn_11294

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*,
_gradient_op_typePartitionedCall-11283*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11282*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input: : : : : : : : 
╬+
ь
while_body_9935
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
┴
х
lstm_1_while_cond_12495
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_12
.lstm_1_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
_output_shapes
: *
T0k
Less_1Lesslstm_1_while_loop_counterlstm_1_while_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::
 :  : : : : : : : : :	 
зG
Ћ
__inference_standard_lstm_14253

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:Z         *
T0B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:         *
T0*
Index0*
shrink_axis_maskІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*'
_output_shapes
:         @*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :ZT
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_14147*
_num_original_outputs*
bodyR
while_body_14148*L
_output_shapes:
8: : : : :         @:         @: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:         Z@*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @░

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         Z@Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*\
_input_shapesK
I:         Z:         @:         @:::*=
api_implements+)lstm_f39847c2-7a35-4198-9536-241cf3e2865c*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
лP
Ї
(__forward_cudnn_lstm_with_fallback_10815

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
dtype0*
value	B :;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB: C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_10634_10816*
_input_shapes *=
api_implements+)lstm_bf9c7feb-2d27-47e7-bd8c-0cf8dbdb35d8*
api_preferred_deviceGPU20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
лP
Ї
(__forward_cudnn_lstm_with_fallback_11785

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
dtype0*
value	B :;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
dtype0*
valueBђ*    ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
dtype0*
valueB:
         E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0o
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
dtype0*
valueB:к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
dtype0*
valueB: E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_11604_11786*
_input_shapes *=
api_implements+)lstm_74cdc0f2-ebae-400d-825a-4e762059078b*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ќ
║
while_body_9631
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5ѕбStatefulPartitionedCallѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @■
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0*+
_gradient_op_typePartitionedCall-9330*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9289*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:         @:         @:         @*
Tin

2ц
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
dtype0*
_output_shapes
: *
value	B :J
addAddV2placeholderadd/y:output:0*
_output_shapes
: *
T0I
add_1/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_1AddV2while_loop_counteradd_1/y:output:0*
T0*
_output_shapes
: Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: ё

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         @ё

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 :
 
лP
Ї
(__forward_cudnn_lstm_with_fallback_14073

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
dtype0*
value	B : H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
dtype0*
valueBђ*    ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
dtype0*
valueB"       L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
dtype0*
valueB"       N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0o
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0I
transpose_9/permConst*
dtype0*!
valueB"          O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"
init_hinit_h_0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_13892_14074*
_input_shapes *=
api_implements+)lstm_af332698-df81-42b0-b14c-2c32c2159570*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
­
ї
+__inference_lstm_cell_1_layer_call_fn_15364

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2ѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*+
_gradient_op_typePartitionedCall-9347*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9324*
Tout
2**
config_proto

GPU 

CPU2J 8*M
_output_shapes;
9:         @:         @:         @*
Tin

2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @ё

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         @ё

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*'
_output_shapes
:         @*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         @:         @:         @:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
ы
ћ
E__inference_sequential_layer_call_and_return_conditional_losses_11282

inputs'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallЪ
lstm/StatefulPartitionedCallStatefulPartitionedCallinputs#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-10819*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_10818*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         Z@─
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_11167*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         @*
Tin
2*,
_gradient_op_typePartitionedCall-11179џ
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11203*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-11204╬
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
Б0
╗	
__inference__traced_save_15452
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop+
'savev2_rmsprop_iter_read_readvariableop	,
(savev2_rmsprop_decay_read_readvariableop4
0savev2_rmsprop_learning_rate_read_readvariableop/
+savev2_rmsprop_momentum_read_readvariableop*
&savev2_rmsprop_rho_read_readvariableop*
&savev2_lstm_kernel_read_readvariableop4
0savev2_lstm_recurrent_kernel_read_readvariableop(
$savev2_lstm_bias_read_readvariableop,
(savev2_lstm_1_kernel_read_readvariableop6
2savev2_lstm_1_recurrent_kernel_read_readvariableop*
&savev2_lstm_1_bias_read_readvariableop7
3savev2_rmsprop_dense_kernel_rms_read_readvariableop5
1savev2_rmsprop_dense_bias_rms_read_readvariableop6
2savev2_rmsprop_lstm_kernel_rms_read_readvariableop@
<savev2_rmsprop_lstm_recurrent_kernel_rms_read_readvariableop4
0savev2_rmsprop_lstm_bias_rms_read_readvariableop8
4savev2_rmsprop_lstm_1_kernel_rms_read_readvariableopB
>savev2_rmsprop_lstm_1_recurrent_kernel_rms_read_readvariableop6
2savev2_rmsprop_lstm_1_bias_rms_read_readvariableop
savev2_1_const

identity_1ѕбMergeV2CheckpointsбSaveV2бSaveV2_1ј
StringJoin/inputs_1Const"/device:CPU:0*<
value3B1 B+_temp_40b5480fed85462db4bef13218aaa962/part*
dtype0*
_output_shapes
: s

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: L

num_shardsConst*
value	B :*
dtype0*
_output_shapes
: f
ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: Њ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ч	
SaveV2/tensor_namesConst"/device:CPU:0*Ц	
valueЏ	Bў	B6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB-optimizer/momentum/.ATTRIBUTES/VARIABLE_VALUEB(optimizer/rho/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEBTlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/0/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/1/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/2/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/3/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/4/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUEBDvariables/5/.OPTIMIZER_SLOT/optimizer/rms/.ATTRIBUTES/VARIABLE_VALUE*
dtype0*
_output_shapes
:Ќ
SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*
_output_shapes
:*=
value4B2B B B B B B B B B B B B B B B B B B B B B ю	
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop'savev2_rmsprop_iter_read_readvariableop(savev2_rmsprop_decay_read_readvariableop0savev2_rmsprop_learning_rate_read_readvariableop+savev2_rmsprop_momentum_read_readvariableop&savev2_rmsprop_rho_read_readvariableop&savev2_lstm_kernel_read_readvariableop0savev2_lstm_recurrent_kernel_read_readvariableop$savev2_lstm_bias_read_readvariableop(savev2_lstm_1_kernel_read_readvariableop2savev2_lstm_1_recurrent_kernel_read_readvariableop&savev2_lstm_1_bias_read_readvariableop3savev2_rmsprop_dense_kernel_rms_read_readvariableop1savev2_rmsprop_dense_bias_rms_read_readvariableop2savev2_rmsprop_lstm_kernel_rms_read_readvariableop<savev2_rmsprop_lstm_recurrent_kernel_rms_read_readvariableop0savev2_rmsprop_lstm_bias_rms_read_readvariableop4savev2_rmsprop_lstm_1_kernel_rms_read_readvariableop>savev2_rmsprop_lstm_1_recurrent_kernel_rms_read_readvariableop2savev2_rmsprop_lstm_1_bias_rms_read_readvariableop"/device:CPU:0*#
dtypes
2	*
_output_shapes
 h
ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: Ќ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: Ѕ
SaveV2_1/tensor_namesConst"/device:CPU:0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH*
dtype0*
_output_shapes
:q
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:├
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
2╣
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
_output_shapes
:*
T0ќ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: s

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0*и
_input_shapesЦ
б: :@:: : : : : :	ђ:	@ђ:ђ:	@ђ:	@ђ:ђ:@::	ђ:	@ђ:ђ:	@ђ:	@ђ:ђ: 2
SaveV2_1SaveV2_12
SaveV2SaveV22(
MergeV2CheckpointsMergeV2Checkpoints: : : : : :+ '
%
_user_specified_namefile_prefix: : : : : : : : :	 :
 : : : : : : : 
О
Ј
while_cond_8809
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::  : : : : : : : : :	 :
 
ќ

ђ
#__inference_signature_wrapper_11313

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallљ
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8*(
f#R!
__inference__wrapped_model_7306*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-11302ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input: : : : : : : : 
═,
Б
lstm_1_while_body_12496
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ћ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         @*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: \
add_3AddV2lstm_1_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Б

Identity_1Identitylstm_1_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"ф
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:
 :  : : : : : : : : :	 
Т-
№
!sequential_lstm_1_while_body_7199(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3'
#sequential_lstm_1_strided_slice_1_0c
_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5%
!sequential_lstm_1_strided_slice_1a
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:а
#TensorArrayV2Read/TensorListGetItemTensorListGetItem_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         @*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: g
add_3AddV2$sequential_lstm_1_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: «

Identity_1Identity*sequential_lstm_1_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"H
!sequential_lstm_1_strided_slice_1#sequential_lstm_1_strided_slice_1_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"└
]tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_tensorarrayv2read_tensorlistgetitem_sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
лP
Ї
(__forward_cudnn_lstm_with_fallback_13584

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
dtype0*
value	B : H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
dtype0*
valueBђ*    ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
dtype0*
valueB"       L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
dtype0*
valueB"       N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"-
transpose_4_permtranspose_4/perm:output:0"!

expanddimsExpandDims:output:0"/
split_2_split_dimsplit_2/split_dim:output:0"%
expanddims_1ExpandDims_1:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"-
transpose_7_permtranspose_7/perm:output:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_13403_13585*
_input_shapes *=
api_implements+)lstm_62400428-7c83-468c-b0ce-181c88f5395f*
api_preferred_deviceGPU2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel
¤+
Ь
while_body_13675
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         @*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         @*
T0K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: :	 :
 :  : : : : : : : 
О
Ј
while_cond_8327
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : :	 :
 :  : : : 
¤+
Ь
while_body_13188
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:         @*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:         @*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
лP
Ї
(__forward_cudnn_lstm_with_fallback_14546

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
dtype0*
value	B : ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB: C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
Index0*
T0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
dtype0*
valueB:к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0"!

identity_1Identity_1:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"-
transpose_8_permtranspose_8/perm:output:0"
identityIdentity:output:0"-
transpose_2_permtranspose_2/perm:output:0"3
strided_slice_stackstrided_slice/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"
	transposetranspose_0:y:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_14365_14547*
_input_shapes *=
api_implements+)lstm_f39847c2-7a35-4198-9536-241cf3e2865c*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel
зG
Ћ
__inference_standard_lstm_10522

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:Z         *
T0B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         *
T0*
Index0І
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0є
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*'
_output_shapes
:         @*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :ZT
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_10417*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_10416*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         @*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:         Z@*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0░

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         Z@Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*\
_input_shapesK
I:         Z:         @:         @:::*=
api_implements+)lstm_bf9c7feb-2d27-47e7-bd8c-0cf8dbdb35d8*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
¤+
Ь
while_body_10417
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*'
_output_shapes
:         @*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         @*
T0K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:         @*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: : : :	 :
 :  : : : : : 
▒
═
&__inference_lstm_1_layer_call_fn_15245
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         @*
Tin
2*+
_gradient_op_typePartitionedCall-9712*I
fDRB
@__inference_lstm_1_layer_call_and_return_conditional_losses_9711ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
§
ў
E__inference_sequential_layer_call_and_return_conditional_losses_11233

lstm_input'
#lstm_statefulpartitionedcall_args_1'
#lstm_statefulpartitionedcall_args_2'
#lstm_statefulpartitionedcall_args_3)
%lstm_1_statefulpartitionedcall_args_1)
%lstm_1_statefulpartitionedcall_args_2)
%lstm_1_statefulpartitionedcall_args_3(
$dense_statefulpartitionedcall_args_1(
$dense_statefulpartitionedcall_args_2
identityѕбdense/StatefulPartitionedCallбlstm/StatefulPartitionedCallбlstm_1/StatefulPartitionedCallБ
lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_input#lstm_statefulpartitionedcall_args_1#lstm_statefulpartitionedcall_args_2#lstm_statefulpartitionedcall_args_3*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         Z@*,
_gradient_op_typePartitionedCall-10819*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_10818─
lstm_1/StatefulPartitionedCallStatefulPartitionedCall%lstm/StatefulPartitionedCall:output:0%lstm_1_statefulpartitionedcall_args_1%lstm_1_statefulpartitionedcall_args_2%lstm_1_statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-11179*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_11167*
Tout
2**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         @*
Tin
2џ
dense/StatefulPartitionedCallStatefulPartitionedCall'lstm_1/StatefulPartitionedCall:output:0$dense_statefulpartitionedcall_args_1$dense_statefulpartitionedcall_args_2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-11204*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11203*
Tout
2╬
IdentityIdentity&dense/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall^lstm/StatefulPartitionedCall^lstm_1/StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::2@
lstm_1/StatefulPartitionedCalllstm_1/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall: : : : : : : :* &
$
_user_specified_name
lstm_input: 
џ
С
?__inference_lstm_layer_call_and_return_conditional_losses_14076

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
dtype0*
_output_shapes
: *
value
B :У_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @╬
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*(
f#R!
__inference_standard_lstm_13780*
Tout	
2**
config_proto

GPU 

CPU2J 8*f
_output_shapesT
R:         @:         Z@:         @:         @: *
Tin

2*,
_gradient_op_typePartitionedCall-13781є
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*+
_output_shapes
:         Z@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:         Z:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
Ѕ
њ
while_cond_14785
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
Ъ>
╗
)__inference_cudnn_lstm_with_fallback_8542

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
dtype0*
valueB:
         C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*
dtype0*!
valueB"          O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
dtype0*
valueB:к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes *=
api_implements+)lstm_f89f659c-75ea-4b17-99d9-8c9723ab031f*
api_preferred_deviceGPU20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
ЕH
Ћ
__inference_standard_lstm_13291

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         *
T0*
Index0І
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
         T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
parallel_iterations *
condR
while_cond_13187*
_num_original_outputs*
bodyR
while_body_13188*L
_output_shapes:
8: : : : :         @:         @: : : : : *K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
T0*
Index0*
shrink_axis_maske
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  @*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @╣

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  @Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*e
_input_shapesT
R:                  :         @:         @:::*=
api_implements+)lstm_62400428-7c83-468c-b0ce-181c88f5395f*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
Ъ>
╗
)__inference_cudnn_lstm_with_fallback_9024

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          @
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
dtype0*
valueBђ*    ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
dtype0*
valueB"       L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
dtype0*
valueB"       N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*
dtype0*!
valueB"          O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
dtype0*
valueB:к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
Index0*
T0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes *=
api_implements+)lstm_0408b657-aecf-41bb-983d-774dc27442ad*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
═P
ї
'__forward_cudnn_lstm_with_fallback_8724

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
N*
T01
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
No
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
dtype0*
valueB:к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"-
transpose_7_permtranspose_7/perm:output:0"+
split_split_dimsplit/split_dim:output:0"#
concat_axisconcat/axis:output:0"-
transpose_1_permtranspose_1/perm:output:0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"
init_hinit_h_0"-
transpose_6_permtranspose_6/perm:output:0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"
concat_1concat_1_0:output:0"-
transpose_9_permtranspose_9/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"-
transpose_3_permtranspose_3/perm:output:0*U
backward_function_name;9__inference___backward_cudnn_lstm_with_fallback_8543_8725*
_input_shapes *=
api_implements+)lstm_f89f659c-75ea-4b17-99d9-8c9723ab031f*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
зG
Ћ
__inference_standard_lstm_12136

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Z         B
ShapeShapetranspose:y:0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         *
T0*
Index0І
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :ZT
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
parallel_iterations *
condR
while_cond_12030*
_num_original_outputs*
bodyR
while_body_12031*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB:
         a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?│
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @░

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         Z@Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*\
_input_shapesK
I:         Z:         @:         @:::*=
api_implements+)lstm_5e66a3d3-3a72-45c8-be76-304c5e537995*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel
Ѕ
њ
while_cond_14616
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
е
╔
$__inference_lstm_layer_call_fn_13603

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-10819*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_10818*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         Z@є
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:         Z@"
identityIdentity:output:0*6
_input_shapes%
#:         Z:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
е
╔
$__inference_lstm_layer_call_fn_13595

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallІ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3**
config_proto

GPU 

CPU2J 8*
Tin
2*+
_output_shapes
:         Z@*,
_gradient_op_typePartitionedCall-10337*H
fCRA
?__inference_lstm_layer_call_and_return_conditional_losses_10336*
Tout
2є
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*+
_output_shapes
:         Z@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:         Z:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
─{
Ѓ
E__inference_sequential_layer_call_and_return_conditional_losses_11959

inputs'
#lstm_statefulpartitionedcall_args_3'
#lstm_statefulpartitionedcall_args_4'
#lstm_statefulpartitionedcall_args_5)
%lstm_1_matmul_readvariableop_resource+
'lstm_1_matmul_1_readvariableop_resource*
&lstm_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource
identityѕбdense/BiasAdd/ReadVariableOpбdense/MatMul/ReadVariableOpбlstm/StatefulPartitionedCallбlstm_1/BiasAdd/ReadVariableOpбlstm_1/MatMul/ReadVariableOpбlstm_1/MatMul_1/ReadVariableOpбlstm_1/while@

lstm/ShapeShapeinputs*
T0*
_output_shapes
:b
lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:d
lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ж
lstm/strided_sliceStridedSlicelstm/Shape:output:0!lstm/strided_slice/stack:output:0#lstm/strided_slice/stack_1:output:0#lstm/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0R
lstm/zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@n
lstm/zeros/mulMullstm/strided_slice:output:0lstm/zeros/mul/y:output:0*
_output_shapes
: *
T0T
lstm/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: h
lstm/zeros/LessLesslstm/zeros/mul:z:0lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: U
lstm/zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: ѓ
lstm/zeros/packedPacklstm/strided_slice:output:0lstm/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:U
lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: {

lstm/zerosFilllstm/zeros/packed:output:0lstm/zeros/Const:output:0*'
_output_shapes
:         @*
T0T
lstm/zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@r
lstm/zeros_1/mulMullstm/strided_slice:output:0lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: V
lstm/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: n
lstm/zeros_1/LessLesslstm/zeros_1/mul:z:0lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: W
lstm/zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@є
lstm/zeros_1/packedPacklstm/strided_slice:output:0lstm/zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0W
lstm/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Ђ
lstm/zeros_1Filllstm/zeros_1/packed:output:0lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @В
lstm/StatefulPartitionedCallStatefulPartitionedCallinputslstm/zeros:output:0lstm/zeros_1:output:0#lstm_statefulpartitionedcall_args_3#lstm_statefulpartitionedcall_args_4#lstm_statefulpartitionedcall_args_5*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*f
_output_shapesT
R:         @:         Z@:         @:         @: *,
_gradient_op_typePartitionedCall-11493*(
f#R!
__inference_standard_lstm_11492a
lstm_1/ShapeShape%lstm/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:d
lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:f
lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:З
lstm_1/strided_sliceStridedSlicelstm_1/Shape:output:0#lstm_1/strided_slice/stack:output:0%lstm_1/strided_slice/stack_1:output:0%lstm_1/strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskT
lstm_1/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: t
lstm_1/zeros/mulMullstm_1/strided_slice:output:0lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: V
lstm_1/zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :Уn
lstm_1/zeros/LessLesslstm_1/zeros/mul:z:0lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: W
lstm_1/zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: ѕ
lstm_1/zeros/packedPacklstm_1/strided_slice:output:0lstm_1/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:W
lstm_1/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    Ђ
lstm_1/zerosFilllstm_1/zeros/packed:output:0lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:         @V
lstm_1/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: x
lstm_1/zeros_1/mulMullstm_1/strided_slice:output:0lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: X
lstm_1/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: t
lstm_1/zeros_1/LessLesslstm_1/zeros_1/mul:z:0lstm_1/zeros_1/Less/y:output:0*
_output_shapes
: *
T0Y
lstm_1/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: ї
lstm_1/zeros_1/packedPacklstm_1/strided_slice:output:0 lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:Y
lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: Є
lstm_1/zeros_1Filllstm_1/zeros_1/packed:output:0lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @j
lstm_1/transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          џ
lstm_1/transpose	Transpose%lstm/StatefulPartitionedCall:output:1lstm_1/transpose/perm:output:0*
T0*+
_output_shapes
:Z         @R
lstm_1/Shape_1Shapelstm_1/transpose:y:0*
T0*
_output_shapes
:f
lstm_1/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:h
lstm_1/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:■
lstm_1/strided_slice_1StridedSlicelstm_1/Shape_1:output:0%lstm_1/strided_slice_1/stack:output:0'lstm_1/strided_slice_1/stack_1:output:0'lstm_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: m
"lstm_1/TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         Х
lstm_1/TensorArrayV2TensorListReserve+lstm_1/TensorArrayV2/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: Ї
<lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Р
.lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorlstm_1/transpose:y:0Elstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: f
lstm_1/strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:h
lstm_1/strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ї
lstm_1/strided_slice_2StridedSlicelstm_1/transpose:y:0%lstm_1/strided_slice_2/stack:output:0'lstm_1/strided_slice_2/stack_1:output:0'lstm_1/strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0▒
lstm_1/MatMul/ReadVariableOpReadVariableOp%lstm_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђЉ
lstm_1/MatMulMatMullstm_1/strided_slice_2:output:0$lstm_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђх
lstm_1/MatMul_1/ReadVariableOpReadVariableOp'lstm_1_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђІ
lstm_1/MatMul_1MatMullstm_1/zeros:output:0&lstm_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђz

lstm_1/addAddV2lstm_1/MatMul:product:0lstm_1/MatMul_1:product:0*(
_output_shapes
:         ђ*
T0»
lstm_1/BiasAdd/ReadVariableOpReadVariableOp&lstm_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђЃ
lstm_1/BiasAddBiasAddlstm_1/add:z:0%lstm_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђN
lstm_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: X
lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: ╦
lstm_1/splitSplitlstm_1/split/split_dim:output:0lstm_1/BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @b
lstm_1/SigmoidSigmoidlstm_1/split:output:0*'
_output_shapes
:         @*
T0d
lstm_1/Sigmoid_1Sigmoidlstm_1/split:output:1*
T0*'
_output_shapes
:         @r

lstm_1/mulMullstm_1/Sigmoid_1:y:0lstm_1/zeros_1:output:0*'
_output_shapes
:         @*
T0\
lstm_1/ReluRelulstm_1/split:output:2*
T0*'
_output_shapes
:         @t
lstm_1/mul_1Mullstm_1/Sigmoid:y:0lstm_1/Relu:activations:0*
T0*'
_output_shapes
:         @i
lstm_1/add_1AddV2lstm_1/mul:z:0lstm_1/mul_1:z:0*
T0*'
_output_shapes
:         @d
lstm_1/Sigmoid_2Sigmoidlstm_1/split:output:3*
T0*'
_output_shapes
:         @Y
lstm_1/Relu_1Relulstm_1/add_1:z:0*
T0*'
_output_shapes
:         @x
lstm_1/mul_2Mullstm_1/Sigmoid_2:y:0lstm_1/Relu_1:activations:0*'
_output_shapes
:         @*
T0u
$lstm_1/TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:║
lstm_1/TensorArrayV2_1TensorListReserve-lstm_1/TensorArrayV2_1/element_shape:output:0lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: M
lstm_1/timeConst*
value	B : *
dtype0*
_output_shapes
: a
lstm_1/while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: [
lstm_1/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: О
lstm_1/whileWhile"lstm_1/while/loop_counter:output:0(lstm_1/while/maximum_iterations:output:0lstm_1/time:output:0lstm_1/TensorArrayV2_1:handle:0lstm_1/zeros:output:0lstm_1/zeros_1:output:0lstm_1/strided_slice_1:output:0>lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:0%lstm_1_matmul_readvariableop_resource'lstm_1_matmul_1_readvariableop_resource&lstm_1_biasadd_readvariableop_resource^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp*#
condR
lstm_1_while_cond_11851*
_num_original_outputs*#
bodyR
lstm_1_while_body_11852*L
_output_shapes:
8: : : : :         @:         @: : : : : *K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations Y
lstm_1/while/IdentityIdentitylstm_1/while:output:0*
T0*
_output_shapes
: [
lstm_1/while/Identity_1Identitylstm_1/while:output:1*
_output_shapes
: *
T0[
lstm_1/while/Identity_2Identitylstm_1/while:output:2*
T0*
_output_shapes
: [
lstm_1/while/Identity_3Identitylstm_1/while:output:3*
_output_shapes
: *
T0l
lstm_1/while/Identity_4Identitylstm_1/while:output:4*
T0*'
_output_shapes
:         @l
lstm_1/while/Identity_5Identitylstm_1/while:output:5*
T0*'
_output_shapes
:         @[
lstm_1/while/Identity_6Identitylstm_1/while:output:6*
T0*
_output_shapes
: [
lstm_1/while/Identity_7Identitylstm_1/while:output:7*
_output_shapes
: *
T0[
lstm_1/while/Identity_8Identitylstm_1/while:output:8*
T0*
_output_shapes
: [
lstm_1/while/Identity_9Identitylstm_1/while:output:9*
T0*
_output_shapes
: ]
lstm_1/while/Identity_10Identitylstm_1/while:output:10*
T0*
_output_shapes
: ѕ
7lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Р
)lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack lstm_1/while/Identity_3:output:0@lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @o
lstm_1/strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:h
lstm_1/strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: h
lstm_1/strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ф
lstm_1/strided_slice_3StridedSlice2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0%lstm_1/strided_slice_3/stack:output:0'lstm_1/strided_slice_3/stack_1:output:0'lstm_1/strided_slice_3/stack_2:output:0*'
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskl
lstm_1/transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          Ф
lstm_1/transpose_1	Transpose2lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0 lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@b
lstm_1/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: «
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@ј
dense/MatMulMatMullstm_1/strided_slice_3:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         г
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:ѕ
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*'
_output_shapes
:         *
T0Е
IdentityIdentitydense/BiasAdd:output:0^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^lstm/StatefulPartitionedCall^lstm_1/BiasAdd/ReadVariableOp^lstm_1/MatMul/ReadVariableOp^lstm_1/MatMul_1/ReadVariableOp^lstm_1/while*
T0*'
_output_shapes
:         "
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2<
lstm/StatefulPartitionedCalllstm/StatefulPartitionedCall2<
lstm_1/MatMul/ReadVariableOplstm_1/MatMul/ReadVariableOp2>
lstm_1/BiasAdd/ReadVariableOplstm_1/BiasAdd/ReadVariableOp2@
lstm_1/MatMul_1/ReadVariableOplstm_1/MatMul_1/ReadVariableOp2
lstm_1/whilelstm_1/while:& "
 
_user_specified_nameinputs: : : : : : : : 
ш─
═
9__inference___backward_cudnn_lstm_with_fallback_6951_7133
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:         @*
T0d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         Z@`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
_output_shapes
:*
T0┤
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:Z         @ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:и
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:Z         @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
_output_shapes
:*
T0─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         @Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
Index0*
T0щ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:Z         @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0»
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*a
_output_shapesO
M:Z         :         @:         @:ђї*
T0ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*+
_output_shapes
:         Z*
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*'
_output_shapes
:         @*
T0^
gradients/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
_output_shapes
: *
T0h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
dtype0*
_output_shapes
:*
valueB:ђ i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:ђ *
Index0*
T0№
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:@*
Index0*
T0№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
_output_shapes
:@*
Index0*
T0m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_5_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	@ђ\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
_output_shapes
: *
T0f
gradients/concat_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:ђh
gradients/concat_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
_output_shapes	
:ђ*
Index0*
T0о
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:         ZЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*B
forward_function_name)'__forward_cudnn_lstm_with_fallback_7132*Ъ
_input_shapesЇ
і:         @:         Z@:         @:         @: :Z         @:::::         @::::         @:::::Z         :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_b2b5e826-16f3-4821-93bd-08aa6fbc38a4*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :  : : : : : : : 
Э
┘
@__inference_dense_layer_call_and_return_conditional_losses_15270

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ѕ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
в+
Ы
while_body_14971
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
dtype0*
_output_shapes
: *
value	B :N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
ЇQ
ц
A__inference_lstm_1_layer_call_and_return_conditional_losses_15070
inputs_0"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@_
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*'
_output_shapes
:         @*
T0O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:x
	transpose	Transposeinputs_0transpose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
T0*
Index0*
shrink_axis_maskБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђ|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         @*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         @*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : c
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
valueB :
         T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
bodyR
while_body_14971*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_14970*
_num_original_outputsK
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*4
_output_shapes"
 :                  @[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *    │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:( $
"
_user_specified_name
inputs/0: : : 
Э─
¤
;__inference___backward_cudnn_lstm_with_fallback_10634_10816
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:         @*
T0d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         Z@`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
_output_shapes
:*
T0┤
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
shrink_axis_mask*+
_output_shapes
:Z         @*
T0*
Index0ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:и
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:Z         @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
Index0*
T0Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
_output_shapes
:*
T0─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
T0*
Index0щ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:Z         @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0»
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*a
_output_shapesO
M:Z         :         @:         @:ђї*
T0ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         Zu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
dtype0*
_output_shapes
:*
valueB:ђ j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
dtype0*
_output_shapes
:*
valueB:@Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ђ №
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_5_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:@*
T0ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes
:	@ђ*
T0\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
_output_shapes
: *
T0f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:         ZЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*'
_output_shapes
:         @*
T0А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*'
_output_shapes
:         @*
T0Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_10815*Ъ
_input_shapesЇ
і:         @:         Z@:         @:         @: :Z         @:::::         @::::         @:::::Z         :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_bf9c7feb-2d27-47e7-bd8c-0cf8dbdb35d8*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
ц
╦
&__inference_lstm_1_layer_call_fn_14895

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallЅ
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*,
_gradient_op_typePartitionedCall-11170*J
fERC
A__inference_lstm_1_layer_call_and_return_conditional_losses_10998*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         @ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*6
_input_shapes%
#:         Z@:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
Ё
љ
while_cond_11386
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
_output_shapes
: *
T0]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::  : : : : : : : : :	 :
 
└B
ь
@__inference_lstm_1_layer_call_and_return_conditional_losses_9711

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCallбwhile;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  @D
Shape_1Shapetranspose:y:0*
_output_shapes
:*
T0_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0Ж
StatefulPartitionedCallStatefulPartitionedCallstrided_slice_2:output:0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*M
_output_shapes;
9:         @:         @:         @*+
_gradient_op_typePartitionedCall-9330*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9289*
Tout
2**
config_proto

GPU 

CPU2J 8n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : ф
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5^StatefulPartitionedCall*
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_9630*
_num_original_outputs*
bodyR
while_body_9631*L
_output_shapes:
8: : : : :         @:         @: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  @*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: ѓ
IdentityIdentitystrided_slice_3:output:0^StatefulPartitionedCall^while*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*?
_input_shapes.
,:                  @:::22
StatefulPartitionedCallStatefulPartitionedCall2
whilewhile: : :& "
 
_user_specified_nameinputs: 
Х
█
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_15303

inputs
states_0
states_1"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identity

identity_1

identity_2ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpБ
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђj
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђp
MatMul_1MatMulstates_0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђА
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @U
mulMulSigmoid_1:y:0states_1*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         @*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         @*
T0ю
IdentityIdentity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_1Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_2Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*X
_input_shapesG
E:         @:         @:         @:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
и

Ѓ
*__inference_sequential_layer_call_fn_12629

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCall▓
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
config_proto

GPU 

CPU2J 8*'
_output_shapes
:         *
Tin
2	*,
_gradient_op_typePartitionedCall-11283*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11282*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*'
_output_shapes
:         *
T0"
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : : : : : : 
┴
х
lstm_1_while_cond_11851
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_lstm_1_strided_slice_12
.lstm_1_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
W
LessLessplaceholderless_lstm_1_strided_slice_1*
T0*
_output_shapes
: k
Less_1Lesslstm_1_while_loop_counterlstm_1_while_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
═,
Б
lstm_1_while_body_11852
lstm_1_while_loop_counter#
lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
lstm_1_strided_slice_1_0X
Ttensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
lstm_1_strided_slice_1V
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ћ
#TensorArrayV2Read/TensorListGetItemTensorListGetItemTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         @*
T0K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: \
add_3AddV2lstm_1_while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Б

Identity_1Identitylstm_1_while_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"ф
Rtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensorTtensorarrayv2read_tensorlistgetitem_lstm_1_tensorarrayunstack_tensorlistfromtensor_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"2
lstm_1_strided_slice_1lstm_1_strided_slice_1_0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp: : : : : :	 :
 :  : : : 
Ё
љ
while_cond_12030
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::
 :  : : : : : : : : :	 
Э─
¤
;__inference___backward_cudnn_lstm_with_fallback_10152_10334
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         Z@`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*'
_output_shapes
:         @*
T0O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:┤
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*+
_output_shapes
:Z         @*
T0*
Index0*
shrink_axis_maskю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:и
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:Z         @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:         @Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*+
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskщ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:Z         @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0»
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:Z         :         @:         @:ђїќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         Zu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
dtype0*
_output_shapes
:*
valueB:ђ j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ђ №
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_2_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_4_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_5_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
_output_shapes
:@*
T0h
gradients/Reshape_9_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_10_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_14_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
N*
_output_shapes
:	ђ*
T0х
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes
:	@ђ*
T0\
gradients/concat_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђ╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:         ZЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
_output_shapes
:	ђ*
T0Њ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
_output_shapes
:	@ђ*
T0Ј

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_10333*Ъ
_input_shapesЇ
і:         @:         Z@:         @:         @: :Z         @:::::         @::::         @:::::Z         :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_4e96b1c4-3fa5-495d-9170-91a62470ff34*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& :  : 
в+
Ы
while_body_14617
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp: :	 :
 :  : : : : : : : 
Э
┘
@__inference_dense_layer_call_and_return_conditional_losses_11203

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpб
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0а
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ѕ
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         @::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs: : 
џ
С
?__inference_lstm_layer_call_and_return_conditional_losses_10336

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         @*
T0╬
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tout	
2**
config_proto

GPU 

CPU2J 8*
Tin

2*f
_output_shapesT
R:         @:         Z@:         @:         @: *,
_gradient_op_typePartitionedCall-10041*(
f#R!
__inference_standard_lstm_10040є
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*+
_output_shapes
:         Z@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:         Z:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
­G
ћ
__inference_standard_lstm_6839

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:Z         *
T0B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:         *
Index0*
T0*
shrink_axis_maskІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:         @*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
dtype0*
_output_shapes
: *
value	B : Z
while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Є
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_6733*
_num_original_outputs*
bodyR
while_body_6734*L
_output_shapes:
8: : : : :         @:         @: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @░

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         Z@Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*\
_input_shapesK
I:         Z:         @:         @:::*=
api_implements+)lstm_b2b5e826-16f3-4821-93bd-08aa6fbc38a4*
api_preferred_deviceCPU2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
а>
╝
*__inference_cudnn_lstm_with_fallback_12247

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          @
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
dtype0*
valueB:
         E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
dtype0*
valueB"       N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
shrink_axis_maskC
strided_slice_2/stackConst*
dtype0*
valueB: E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
dtype0*
valueB:к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_5e66a3d3-3a72-45c8-be76-304c5e537995*
api_preferred_deviceGPU2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
в+
Ы
while_body_14786
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         @*
T0N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
Ё
љ
while_cond_10416
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
­
ї
+__inference_lstm_cell_1_layer_call_fn_15350

inputs
states_0
states_1"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identity

identity_1

identity_2ѕбStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallinputsstates_0states_1statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*+
_gradient_op_typePartitionedCall-9330*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9289*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         @:         @:         @ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         @ё

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         @ё

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*X
_input_shapesG
E:         @:         @:         @:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs:($
"
_user_specified_name
states/0:($
"
_user_specified_name
states/1: : : 
¤+
Ь
while_body_14148
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*'
_output_shapes
:         @*
T0K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"!

identity_1Identity_1:output:0" 
strided_slicestrided_slice_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : : : :	 :
 :  : 
џ
С
?__inference_lstm_layer_call_and_return_conditional_losses_10818

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall;
ShapeShapeinputs*
_output_shapes
:*
T0]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
_output_shapes
: *
T0P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
dtype0*
_output_shapes
: *
value
B :У_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @╬
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*f
_output_shapesT
R:         @:         Z@:         @:         @: *
Tin

2*,
_gradient_op_typePartitionedCall-10523*(
f#R!
__inference_standard_lstm_10522*
Tout	
2**
config_proto

GPU 

CPU2J 8є
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*+
_output_shapes
:         Z@*
T0"
identityIdentity:output:0*6
_input_shapes%
#:         Z:::22
StatefulPartitionedCallStatefulPartitionedCall: : :& "
 
_user_specified_nameinputs: 
пP
б
A__inference_lstm_1_layer_call_and_return_conditional_losses_14718

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
N*
_output_shapes
:*
T0R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         @*
T0c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Z         @D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         @Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђ|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @]
mulMulSigmoid_1:y:0zeros_1:output:0*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*
T0*'
_output_shapes
:         @_
mul_1MulSigmoid:y:0Relu:activations:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*'
_output_shapes
:         @*
T0c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :ZT
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_14617*L
_output_shapes:
8: : : : :         @:         @: : : : : *K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_14616K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         @*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
_output_shapes
: *
T0M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
Index0*
T0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*6
_input_shapes%
#:         Z@:::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs: : : 
К
╦
$__inference_lstm_layer_call_fn_12645
inputs_0"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3
identityѕбStatefulPartitionedCallћ
StatefulPartitionedCallStatefulPartitionedCallinputs_0statefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3*
Tin
2*4
_output_shapes"
 :                  @*+
_gradient_op_typePartitionedCall-9210*G
fBR@
>__inference_lstm_layer_call_and_return_conditional_losses_9209*
Tout
2**
config_proto

GPU 

CPU2J 8Ј
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*?
_input_shapes.
,:                  :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
в+
Ы
while_body_10897
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0Б
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
ReluRelusplit:output:2*'
_output_shapes
:         @*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*'
_output_shapes
:         @*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : :	 :
 :  : : : : : 
дH
ћ
__inference_standard_lstm_8913

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:v
	transpose	Transposeinputstranspose/perm:output:0*
T0*4
_output_shapes"
 :                  B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskf
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*'
_output_shapes
:         І
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Ќ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: c
while/maximum_iterationsConst*
valueB :
         *
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Є
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_8810*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_8809K
while/IdentityIdentitywhile:output:0*
_output_shapes
: *
T0M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:о
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*4
_output_shapes"
 :                  @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:Ъ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*4
_output_shapes"
 :                  @*
T0[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @╣

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*4
_output_shapes"
 :                  @Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*e
_input_shapesT
R:                  :         @:         @:::*=
api_implements+)lstm_0408b657-aecf-41bb-983d-774dc27442ad*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
џ
С
?__inference_lstm_layer_call_and_return_conditional_losses_14549

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
dtype0*
_output_shapes
: *
value	B :@c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @╬
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*,
_gradient_op_typePartitionedCall-14254*(
f#R!
__inference_standard_lstm_14253*
Tout	
2**
config_proto

GPU 

CPU2J 8*f
_output_shapesT
R:         @:         Z@:         @:         @: *
Tin

2є
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*+
_output_shapes
:         Z@"
identityIdentity:output:0*6
_input_shapes%
#:         Z:::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
Ў
в
!sequential_lstm_1_while_cond_7198(
$sequential_lstm_1_while_loop_counter.
*sequential_lstm_1_while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3*
&less_sequential_lstm_1_strided_slice_1=
9sequential_lstm_1_tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
b
LessLessplaceholder&less_sequential_lstm_1_strided_slice_1*
T0*
_output_shapes
: Ђ
Less_1Less$sequential_lstm_1_while_loop_counter*sequential_lstm_1_while_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : :	 :
 :  : : : : : 
├

Є
*__inference_sequential_layer_call_fn_11263

lstm_input"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8
identityѕбStatefulPartitionedCallХ
StatefulPartitionedCallStatefulPartitionedCall
lstm_inputstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8**
config_proto

GPU 

CPU2J 8*
Tin
2	*'
_output_shapes
:         *,
_gradient_op_typePartitionedCall-11252*N
fIRG
E__inference_sequential_layer_call_and_return_conditional_losses_11251*
Tout
2ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
lstm_input: : : : : : : : 
зG
Ћ
__inference_standard_lstm_11492

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*
dtype0*
_output_shapes
:*!
valueB"          m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:Z         *
T0B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:         *
T0*
Index0*
shrink_axis_maskІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЌ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0є
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: T
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_num_original_outputs*
bodyR
while_body_11387*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_11386K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
_output_shapes
: *
T0M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         @*
T0M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
_output_shapes
: *
T0M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
dtype0*
_output_shapes
: *
valueB
 *  ђ?│
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0░

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         Z@Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*\
_input_shapesK
I:         Z:         @:         @:::*=
api_implements+)lstm_74cdc0f2-ebae-400d-825a-4e762059078b*
api_preferred_deviceCPU20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
¤+
Ь
while_body_11387
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"       ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:
 :  : : : : : : : : :	 
Ѕ
њ
while_cond_11065
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
_output_shapes
: *
T0
"
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : :	 :
 :  : : : : : 
╬+
ь
while_body_6734
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"       ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*'
_output_shapes
:         @*
T0Z
mulMulSigmoid_1:y:0placeholder_3*'
_output_shapes
:         @*
T0N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:  : : : : : : : : :	 :
 
зG
Ћ
__inference_standard_lstm_13780

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Z         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0f
TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         *
Index0*
T0І
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Ќ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*'
_output_shapes
:         @*
T0Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:         @*
T0n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: T
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Ѕ
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
condR
while_cond_13674*
_num_original_outputs*
bodyR
while_body_13675*L
_output_shapes:
8: : : : :         @:         @: : : : : *K
output_shapes:
8: : : : :         @:         @: : : : : *
T
2*
_lower_using_switch_merge(*
parallel_iterations K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
_output_shapes
: *
T0^
while/Identity_4Identitywhile:output:4*'
_output_shapes
:         @*
T0^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
_output_shapes
: *
T0O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*!
valueB"          *
dtype0*
_output_shapes
:ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @░

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*+
_output_shapes
:         Z@Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*\
_input_shapesK
I:         Z:         @:         @:::*=
api_implements+)lstm_af332698-df81-42b0-b14c-2c32c2159570*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
й
Т
?__inference_lstm_layer_call_and_return_conditional_losses_13116
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
T0*
Index0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :УY

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         @*
T0┘
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*(
f#R!
__inference_standard_lstm_12820*
Tout	
2**
config_proto

GPU 

CPU2J 8*o
_output_shapes]
[:         @:                  @:         @:         @: *
Tin

2*,
_gradient_op_typePartitionedCall-12821Ј
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*?
_input_shapes.
,:                  :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
ё
Ј
while_cond_9934
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
_output_shapes
: *
T0]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
_output_shapes
: *
T0F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::
 :  : : : : : : : : :	 
а>
╝
*__inference_cudnn_lstm_with_fallback_10633

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*
dtype0*!
valueB"          @
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
dtype0*
value	B :i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
dtype0*
value	B :u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
	num_split*
T0>
Const_3Const*
dtype0*
valueB:
         E
transpose_1/permConst*
dtype0*
valueB"       L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0k
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
dtype0*
valueB:╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*
_input_shapes *=
api_implements+)lstm_bf9c7feb-2d27-47e7-bd8c-0cf8dbdb35d8*
api_preferred_deviceGPU2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
ЂЊ
є
__inference__wrapped_model_7306

lstm_input2
.sequential_lstm_statefulpartitionedcall_args_32
.sequential_lstm_statefulpartitionedcall_args_42
.sequential_lstm_statefulpartitionedcall_args_54
0sequential_lstm_1_matmul_readvariableop_resource6
2sequential_lstm_1_matmul_1_readvariableop_resource5
1sequential_lstm_1_biasadd_readvariableop_resource3
/sequential_dense_matmul_readvariableop_resource4
0sequential_dense_biasadd_readvariableop_resource
identityѕб'sequential/dense/BiasAdd/ReadVariableOpб&sequential/dense/MatMul/ReadVariableOpб'sequential/lstm/StatefulPartitionedCallб(sequential/lstm_1/BiasAdd/ReadVariableOpб'sequential/lstm_1/MatMul/ReadVariableOpб)sequential/lstm_1/MatMul_1/ReadVariableOpбsequential/lstm_1/whileO
sequential/lstm/ShapeShape
lstm_input*
T0*
_output_shapes
:m
#sequential/lstm/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:o
%sequential/lstm/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:o
%sequential/lstm/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:А
sequential/lstm/strided_sliceStridedSlicesequential/lstm/Shape:output:0,sequential/lstm/strided_slice/stack:output:0.sequential/lstm/strided_slice/stack_1:output:0.sequential/lstm/strided_slice/stack_2:output:0*
T0*
Index0*
shrink_axis_mask*
_output_shapes
: ]
sequential/lstm/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: Ј
sequential/lstm/zeros/mulMul&sequential/lstm/strided_slice:output:0$sequential/lstm/zeros/mul/y:output:0*
T0*
_output_shapes
: _
sequential/lstm/zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Ѕ
sequential/lstm/zeros/LessLesssequential/lstm/zeros/mul:z:0%sequential/lstm/zeros/Less/y:output:0*
T0*
_output_shapes
: `
sequential/lstm/zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: Б
sequential/lstm/zeros/packedPack&sequential/lstm/strided_slice:output:0'sequential/lstm/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:`
sequential/lstm/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: ю
sequential/lstm/zerosFill%sequential/lstm/zeros/packed:output:0$sequential/lstm/zeros/Const:output:0*
T0*'
_output_shapes
:         @_
sequential/lstm/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: Њ
sequential/lstm/zeros_1/mulMul&sequential/lstm/strided_slice:output:0&sequential/lstm/zeros_1/mul/y:output:0*
T0*
_output_shapes
: a
sequential/lstm/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Ј
sequential/lstm/zeros_1/LessLesssequential/lstm/zeros_1/mul:z:0'sequential/lstm/zeros_1/Less/y:output:0*
T0*
_output_shapes
: b
 sequential/lstm/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: Д
sequential/lstm/zeros_1/packedPack&sequential/lstm/strided_slice:output:0)sequential/lstm/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:b
sequential/lstm/zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    б
sequential/lstm/zeros_1Fill'sequential/lstm/zeros_1/packed:output:0&sequential/lstm/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @░
'sequential/lstm/StatefulPartitionedCallStatefulPartitionedCall
lstm_inputsequential/lstm/zeros:output:0 sequential/lstm/zeros_1:output:0.sequential_lstm_statefulpartitionedcall_args_3.sequential_lstm_statefulpartitionedcall_args_4.sequential_lstm_statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*
Tin

2*f
_output_shapesT
R:         @:         Z@:         @:         @: *+
_gradient_op_typePartitionedCall-6840*'
f"R 
__inference_standard_lstm_6839*
Tout	
2w
sequential/lstm_1/ShapeShape0sequential/lstm/StatefulPartitionedCall:output:1*
T0*
_output_shapes
:o
%sequential/lstm_1/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:q
'sequential/lstm_1/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:q
'sequential/lstm_1/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Ф
sequential/lstm_1/strided_sliceStridedSlice sequential/lstm_1/Shape:output:0.sequential/lstm_1/strided_slice/stack:output:00sequential/lstm_1/strided_slice/stack_1:output:00sequential/lstm_1/strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
Index0*
T0_
sequential/lstm_1/zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: Ћ
sequential/lstm_1/zeros/mulMul(sequential/lstm_1/strided_slice:output:0&sequential/lstm_1/zeros/mul/y:output:0*
T0*
_output_shapes
: a
sequential/lstm_1/zeros/Less/yConst*
dtype0*
_output_shapes
: *
value
B :УЈ
sequential/lstm_1/zeros/LessLesssequential/lstm_1/zeros/mul:z:0'sequential/lstm_1/zeros/Less/y:output:0*
T0*
_output_shapes
: b
 sequential/lstm_1/zeros/packed/1Const*
dtype0*
_output_shapes
: *
value	B :@Е
sequential/lstm_1/zeros/packedPack(sequential/lstm_1/strided_slice:output:0)sequential/lstm_1/zeros/packed/1:output:0*
T0*
N*
_output_shapes
:b
sequential/lstm_1/zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: б
sequential/lstm_1/zerosFill'sequential/lstm_1/zeros/packed:output:0&sequential/lstm_1/zeros/Const:output:0*
T0*'
_output_shapes
:         @a
sequential/lstm_1/zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: Ў
sequential/lstm_1/zeros_1/mulMul(sequential/lstm_1/strided_slice:output:0(sequential/lstm_1/zeros_1/mul/y:output:0*
T0*
_output_shapes
: c
 sequential/lstm_1/zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Ћ
sequential/lstm_1/zeros_1/LessLess!sequential/lstm_1/zeros_1/mul:z:0)sequential/lstm_1/zeros_1/Less/y:output:0*
T0*
_output_shapes
: d
"sequential/lstm_1/zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: Г
 sequential/lstm_1/zeros_1/packedPack(sequential/lstm_1/strided_slice:output:0+sequential/lstm_1/zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:d
sequential/lstm_1/zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: е
sequential/lstm_1/zeros_1Fill)sequential/lstm_1/zeros_1/packed:output:0(sequential/lstm_1/zeros_1/Const:output:0*
T0*'
_output_shapes
:         @u
 sequential/lstm_1/transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:╗
sequential/lstm_1/transpose	Transpose0sequential/lstm/StatefulPartitionedCall:output:1)sequential/lstm_1/transpose/perm:output:0*+
_output_shapes
:Z         @*
T0h
sequential/lstm_1/Shape_1Shapesequential/lstm_1/transpose:y:0*
_output_shapes
:*
T0q
'sequential/lstm_1/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: s
)sequential/lstm_1/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:х
!sequential/lstm_1/strided_slice_1StridedSlice"sequential/lstm_1/Shape_1:output:00sequential/lstm_1/strided_slice_1/stack:output:02sequential/lstm_1/strided_slice_1/stack_1:output:02sequential/lstm_1/strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0x
-sequential/lstm_1/TensorArrayV2/element_shapeConst*
valueB :
         *
dtype0*
_output_shapes
: О
sequential/lstm_1/TensorArrayV2TensorListReserve6sequential/lstm_1/TensorArrayV2/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: ў
Gsequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ѓ
9sequential/lstm_1/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorsequential/lstm_1/transpose:y:0Psequential/lstm_1/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: q
'sequential/lstm_1/strided_slice_2/stackConst*
dtype0*
_output_shapes
:*
valueB: s
)sequential/lstm_1/strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_2/stack_2Const*
dtype0*
_output_shapes
:*
valueB:├
!sequential/lstm_1/strided_slice_2StridedSlicesequential/lstm_1/transpose:y:00sequential/lstm_1/strided_slice_2/stack:output:02sequential/lstm_1/strided_slice_2/stack_1:output:02sequential/lstm_1/strided_slice_2/stack_2:output:0*'
_output_shapes
:         @*
T0*
Index0*
shrink_axis_maskК
'sequential/lstm_1/MatMul/ReadVariableOpReadVariableOp0sequential_lstm_1_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђ▓
sequential/lstm_1/MatMulMatMul*sequential/lstm_1/strided_slice_2:output:0/sequential/lstm_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђ╦
)sequential/lstm_1/MatMul_1/ReadVariableOpReadVariableOp2sequential_lstm_1_matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђг
sequential/lstm_1/MatMul_1MatMul sequential/lstm_1/zeros:output:01sequential/lstm_1/MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЏ
sequential/lstm_1/addAddV2"sequential/lstm_1/MatMul:product:0$sequential/lstm_1/MatMul_1:product:0*
T0*(
_output_shapes
:         ђ┼
(sequential/lstm_1/BiasAdd/ReadVariableOpReadVariableOp1sequential_lstm_1_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђц
sequential/lstm_1/BiasAddBiasAddsequential/lstm_1/add:z:00sequential/lstm_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђY
sequential/lstm_1/ConstConst*
value	B :*
dtype0*
_output_shapes
: c
!sequential/lstm_1/split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: В
sequential/lstm_1/splitSplit*sequential/lstm_1/split/split_dim:output:0"sequential/lstm_1/BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @x
sequential/lstm_1/SigmoidSigmoid sequential/lstm_1/split:output:0*
T0*'
_output_shapes
:         @z
sequential/lstm_1/Sigmoid_1Sigmoid sequential/lstm_1/split:output:1*
T0*'
_output_shapes
:         @Њ
sequential/lstm_1/mulMulsequential/lstm_1/Sigmoid_1:y:0"sequential/lstm_1/zeros_1:output:0*'
_output_shapes
:         @*
T0r
sequential/lstm_1/ReluRelu sequential/lstm_1/split:output:2*
T0*'
_output_shapes
:         @Ћ
sequential/lstm_1/mul_1Mulsequential/lstm_1/Sigmoid:y:0$sequential/lstm_1/Relu:activations:0*'
_output_shapes
:         @*
T0і
sequential/lstm_1/add_1AddV2sequential/lstm_1/mul:z:0sequential/lstm_1/mul_1:z:0*
T0*'
_output_shapes
:         @z
sequential/lstm_1/Sigmoid_2Sigmoid sequential/lstm_1/split:output:3*
T0*'
_output_shapes
:         @o
sequential/lstm_1/Relu_1Relusequential/lstm_1/add_1:z:0*'
_output_shapes
:         @*
T0Ў
sequential/lstm_1/mul_2Mulsequential/lstm_1/Sigmoid_2:y:0&sequential/lstm_1/Relu_1:activations:0*
T0*'
_output_shapes
:         @ђ
/sequential/lstm_1/TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:█
!sequential/lstm_1/TensorArrayV2_1TensorListReserve8sequential/lstm_1/TensorArrayV2_1/element_shape:output:0*sequential/lstm_1/strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: X
sequential/lstm_1/timeConst*
dtype0*
_output_shapes
: *
value	B : l
*sequential/lstm_1/while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: f
$sequential/lstm_1/while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: љ
sequential/lstm_1/whileWhile-sequential/lstm_1/while/loop_counter:output:03sequential/lstm_1/while/maximum_iterations:output:0sequential/lstm_1/time:output:0*sequential/lstm_1/TensorArrayV2_1:handle:0 sequential/lstm_1/zeros:output:0"sequential/lstm_1/zeros_1:output:0*sequential/lstm_1/strided_slice_1:output:0Isequential/lstm_1/TensorArrayUnstack/TensorListFromTensor:output_handle:00sequential_lstm_1_matmul_readvariableop_resource2sequential_lstm_1_matmul_1_readvariableop_resource1sequential_lstm_1_biasadd_readvariableop_resource)^sequential/lstm_1/BiasAdd/ReadVariableOp(^sequential/lstm_1/MatMul/ReadVariableOp*^sequential/lstm_1/MatMul_1/ReadVariableOp*
_num_original_outputs*-
body%R#
!sequential_lstm_1_while_body_7199*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *-
cond%R#
!sequential_lstm_1_while_cond_7198o
 sequential/lstm_1/while/IdentityIdentity sequential/lstm_1/while:output:0*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_1Identity sequential/lstm_1/while:output:1*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_2Identity sequential/lstm_1/while:output:2*
_output_shapes
: *
T0q
"sequential/lstm_1/while/Identity_3Identity sequential/lstm_1/while:output:3*
T0*
_output_shapes
: ѓ
"sequential/lstm_1/while/Identity_4Identity sequential/lstm_1/while:output:4*
T0*'
_output_shapes
:         @ѓ
"sequential/lstm_1/while/Identity_5Identity sequential/lstm_1/while:output:5*
T0*'
_output_shapes
:         @q
"sequential/lstm_1/while/Identity_6Identity sequential/lstm_1/while:output:6*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_7Identity sequential/lstm_1/while:output:7*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_8Identity sequential/lstm_1/while:output:8*
T0*
_output_shapes
: q
"sequential/lstm_1/while/Identity_9Identity sequential/lstm_1/while:output:9*
T0*
_output_shapes
: s
#sequential/lstm_1/while/Identity_10Identity!sequential/lstm_1/while:output:10*
T0*
_output_shapes
: Њ
Bsequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ѓ
4sequential/lstm_1/TensorArrayV2Stack/TensorListStackTensorListStack+sequential/lstm_1/while/Identity_3:output:0Ksequential/lstm_1/TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @z
'sequential/lstm_1/strided_slice_3/stackConst*
dtype0*
_output_shapes
:*
valueB:
         s
)sequential/lstm_1/strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:s
)sequential/lstm_1/strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:р
!sequential/lstm_1/strided_slice_3StridedSlice=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:00sequential/lstm_1/strided_slice_3/stack:output:02sequential/lstm_1/strided_slice_3/stack_1:output:02sequential/lstm_1/strided_slice_3/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
Index0*
T0w
"sequential/lstm_1/transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          ╠
sequential/lstm_1/transpose_1	Transpose=sequential/lstm_1/TensorArrayV2Stack/TensorListStack:tensor:0+sequential/lstm_1/transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@m
sequential/lstm_1/runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: ─
&sequential/dense/MatMul/ReadVariableOpReadVariableOp/sequential_dense_matmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes

:@»
sequential/dense/MatMulMatMul*sequential/lstm_1/strided_slice_3:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*'
_output_shapes
:         *
T0┬
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp0sequential_dense_biasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:Е
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Ђ
IdentityIdentity!sequential/dense/BiasAdd:output:0(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp(^sequential/lstm/StatefulPartitionedCall)^sequential/lstm_1/BiasAdd/ReadVariableOp(^sequential/lstm_1/MatMul/ReadVariableOp*^sequential/lstm_1/MatMul_1/ReadVariableOp^sequential/lstm_1/while*'
_output_shapes
:         *
T0"
identityIdentity:output:0*J
_input_shapes9
7:         Z::::::::2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/lstm_1/MatMul_1/ReadVariableOp)sequential/lstm_1/MatMul_1/ReadVariableOp22
sequential/lstm_1/whilesequential/lstm_1/while2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2T
(sequential/lstm_1/BiasAdd/ReadVariableOp(sequential/lstm_1/BiasAdd/ReadVariableOp2R
'sequential/lstm/StatefulPartitionedCall'sequential/lstm/StatefulPartitionedCall2R
'sequential/lstm_1/MatMul/ReadVariableOp'sequential/lstm_1/MatMul/ReadVariableOp: : : : : : : :* &
$
_user_specified_name
lstm_input: 
Ќ
║
while_body_9771
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_1_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 statefulpartitionedcall_args_3_0$
 statefulpartitionedcall_args_4_0$
 statefulpartitionedcall_args_5_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_slice_1O
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5ѕбStatefulPartitionedCallѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         @■
StatefulPartitionedCallStatefulPartitionedCall*TensorArrayV2Read/TensorListGetItem:item:0placeholder_2placeholder_3 statefulpartitionedcall_args_3_0 statefulpartitionedcall_args_4_0 statefulpartitionedcall_args_5_0**
config_proto

GPU 

CPU2J 8*
Tin

2*M
_output_shapes;
9:         @:         @:         @*+
_gradient_op_typePartitionedCall-9347*N
fIRG
E__inference_lstm_cell_1_layer_call_and_return_conditional_losses_9324*
Tout
2ц
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder StatefulPartitionedCall:output:0*
element_dtype0*
_output_shapes
: G
add/yConst*
value	B :*
dtype0*
_output_shapes
: J
addAddV2placeholderadd/y:output:0*
T0*
_output_shapes
: I
add_1/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_1AddV2while_loop_counteradd_1/y:output:0*
_output_shapes
: *
T0Z
IdentityIdentity	add_1:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: k

Identity_1Identitywhile_maximum_iterations^StatefulPartitionedCall*
T0*
_output_shapes
: Z

Identity_2Identityadd:z:0^StatefulPartitionedCall*
T0*
_output_shapes
: Є

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^StatefulPartitionedCall*
T0*
_output_shapes
: ё

Identity_4Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:         @ё

Identity_5Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:         @"B
statefulpartitionedcall_args_4 statefulpartitionedcall_args_4_0"B
statefulpartitionedcall_args_5 statefulpartitionedcall_args_5_0"$
strided_slice_1strided_slice_1_0"!

identity_1Identity_1:output:0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"!

identity_5Identity_5:output:0"
identityIdentity:output:0"B
statefulpartitionedcall_args_3 statefulpartitionedcall_args_3_0*Q
_input_shapes@
>: : : : :         @:         @: : :::22
StatefulPartitionedCallStatefulPartitionedCall:  : : : : : : : : :	 :
 
пP
б
A__inference_lstm_1_layer_call_and_return_conditional_losses_14887

inputs"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resource
identityѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhile;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
_output_shapes
: *
Index0*
T0*
shrink_axis_maskM
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
_output_shapes
: *
T0O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
_output_shapes
: *
T0R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @c
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*+
_output_shapes
:Z         @*
T0D
Shape_1Shapetranspose:y:0*
T0*
_output_shapes
:_
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
dtype0*
_output_shapes
:*
valueB:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:█
strided_slice_1StridedSliceShape_1:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         А
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_2/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:ж
strided_slice_2StridedSlicetranspose:y:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*'
_output_shapes
:         @*
T0*
Index0Б
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђ|
MatMulMatMulstrided_slice_2:output:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђД
MatMul_1/ReadVariableOpReadVariableOp matmul_1_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђv
MatMul_1MatMulzeros:output:0MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*(
_output_shapes
:         ђ*
T0А
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @*
T0T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @]
mulMulSigmoid_1:y:0zeros_1:output:0*'
_output_shapes
:         @*
T0N
ReluRelusplit:output:2*'
_output_shapes
:         @*
T0_
mul_1MulSigmoid:y:0Relu:activations:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Relu_1Relu	add_1:z:0*
T0*'
_output_shapes
:         @c
mul_2MulSigmoid_2:y:0Relu_1:activations:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Ц
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice_1:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
value	B :Z*
dtype0*
_output_shapes
: T
while/loop_counterConst*
dtype0*
_output_shapes
: *
value	B : Я
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0zeros:output:0zeros_1:output:0strided_slice_1:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0matmul_readvariableop_resource matmul_1_readvariableop_resourcebiasadd_readvariableop_resource^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(*
parallel_iterations *
condR
while_cond_14785*
_num_original_outputs*
bodyR
while_body_14786*L
_output_shapes:
8: : : : :         @:         @: : : : : K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
_output_shapes
: *
T0M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*'
_output_shapes
:         @*
T0M
while/Identity_6Identitywhile:output:6*
_output_shapes
: *
T0M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
T0*
_output_shapes
: Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_3/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_3/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_3/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Є
strided_slice_3StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*+
_output_shapes
:         Z@*
T0[
runtimeConst"/device:CPU:0*
valueB
 *    *
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_3:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*6
_input_shapes%
#:         Z@:::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile: : :& "
 
_user_specified_nameinputs: 
а>
╝
*__inference_cudnn_lstm_with_fallback_13891

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
dtype0*
valueBђ*    ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
dtype0*
valueB"       N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
dtype0*
valueB"       N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
valueB: *
dtype0C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
strided_slice_2/stackConst*
dtype0*
valueB: E
strided_slice_2/stack_1Const*
dtype0*
valueB:E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes *=
api_implements+)lstm_af332698-df81-42b0-b14c-2c32c2159570*
api_preferred_deviceGPU26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel
Ъ>
╗
)__inference_cudnn_lstm_with_fallback_6950

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0@
	transpose	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
value	B : *
dtype0B

ExpandDims
ExpandDimsinit_hExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0F
ExpandDims_1
ExpandDimsinit_cExpandDims_1/dim:output:0*
T0/
ConstConst*
value	B :*
dtype09
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
T0*
	num_split1
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
T0*
	num_splitl
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
dtype0*
value	B : n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
value	B :*
dtype0;
split_2/split_dimConst*
value	B : *
dtype0W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
valueB"       *
dtype0L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
valueB"       *
dtype0N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
valueB"       *
dtype0N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ѓ
concat_1ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
T0*
Nk
CudnnRNNCudnnRNNtranspose:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB: C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
T0*
Index0I
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
valueB:*
dtype0E
strided_slice_1/stack_2Const*
dtype0*
valueB:к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
dtype0*
valueB:к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
runtimeConst"/device:CPU:0*
valueB
 *   @*
dtype0ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*
_input_shapes *=
api_implements+)lstm_b2b5e826-16f3-4821-93bd-08aa6fbc38a4*
api_preferred_deviceGPU20
split_1/ReadVariableOpsplit_1/ReadVariableOp2
CudnnRNNCudnnRNN2,
split/ReadVariableOpsplit/ReadVariableOp2.
concat/ReadVariableOpconcat/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
Э─
¤
;__inference___backward_cudnn_lstm_with_fallback_11604_11786
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:         @*
T0d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         Z@`
gradients/grad_ys_2Identityplaceholder_2*'
_output_shapes
:         @*
T0`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
T0*
_output_shapes
: }
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
_output_shapes
:*
T0┤
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:Z         @ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:и
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:Z         @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:         @Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
T0*
Index0щ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:Z         @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:»
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*a
_output_shapesO
M:Z         :         @:         @:ђї*
T0ќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*+
_output_shapes
:         Z*
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*'
_output_shapes
:         @*
T0y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
_output_shapes
: *
T0h
gradients/concat_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_4Const*
dtype0*
_output_shapes
:*
valueB:ђ j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
dtype0*
_output_shapes
:*
valueB:ђ j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
dtype0*
_output_shapes
:*
valueB:@i
gradients/concat_1_grad/Shape_9Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_12Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_13Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_14Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
dtype0*
_output_shapes
:*
valueB:@Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ђ №
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:@*
Index0*
T0№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@      А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_5_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_10_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
_output_shapes
:@*
T0ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
N*
_output_shapes
:	@ђ*
T0\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
_output_shapes
: *
T0f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђ╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:         ZЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*'
_output_shapes
:         @*
T0А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_5Identity_5:output:0"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_11785*Ъ
_input_shapesЇ
і:         @:         Z@:         @:         @: :Z         @:::::         @::::         @:::::Z         :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_74cdc0f2-ebae-400d-825a-4e762059078b*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop: : : : : : :  :! :" :# :$ :% :& :  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : 
й
Т
?__inference_lstm_layer_call_and_return_conditional_losses_13587
inputs_0"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall=
ShapeShapeinputs_0*
_output_shapes
:*
T0]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
N*
_output_shapes
:*
T0P
zeros/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
T0*
_output_shapes
: Q
zeros_1/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: _
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*'
_output_shapes
:         @*
T0┘
StatefulPartitionedCallStatefulPartitionedCallinputs_0zeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5**
config_proto

GPU 

CPU2J 8*
Tin

2*o
_output_shapes]
[:         @:                  @:         @:         @: *,
_gradient_op_typePartitionedCall-13292*(
f#R!
__inference_standard_lstm_13291*
Tout	
2Ј
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*?
_input_shapes.
,:                  :::22
StatefulPartitionedCallStatefulPartitionedCall:( $
"
_user_specified_name
inputs/0: : : 
лP
Ї
(__forward_cudnn_lstm_with_fallback_13113

inputs
init_h_0
init_c_0

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4
cudnnrnn
strided_slice_stack
strided_slice_stack_1
strided_slice_stack_2
transpose_9_perm

cudnnrnn_0
strided_slice_1_stack
strided_slice_1_stack_1
strided_slice_1_stack_2

cudnnrnn_1
strided_slice_2_stack
strided_slice_2_stack_1
strided_slice_2_stack_2

cudnnrnn_2
	transpose

expanddims
expanddims_1
concat_1
transpose_perm

init_h

init_c
concat_1_axis
transpose_1_perm
transpose_2_perm
transpose_3_perm
transpose_4_perm
transpose_5_perm
transpose_6_perm
transpose_7_perm
transpose_8_perm
split_2_split_dim
split_split_dim
split_1_split_dim
concat_axisѕбCudnnRNNбconcat/ReadVariableOpбsplit/ReadVariableOpбsplit_1/ReadVariableOpбzeros_like/ReadVariableOpG
transpose/permConst*!
valueB"          *
dtype0B
transpose_0	Transposeinputstranspose/perm:output:0*
T08
ExpandDims/dimConst*
dtype0*
value	B : D

ExpandDims
ExpandDimsinit_h_0ExpandDims/dim:output:0*
T0:
ExpandDims_1/dimConst*
value	B : *
dtype0H
ExpandDims_1
ExpandDimsinit_c_0ExpandDims_1/dim:output:0*
T0/
ConstConst*
dtype0*
value	B :9
split/split_dimConst*
value	B :*
dtype0i
split/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0`
splitSplitsplit/split_dim:output:0split/ReadVariableOp:value:0*
	num_split*
T01
Const_1Const*
value	B :*
dtype0;
split_1/split_dimConst*
value	B :*
dtype0u
split_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0f
split_1Splitsplit_1/split_dim:output:0split_1/ReadVariableOp:value:0*
	num_split*
T0l
zeros_like/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0<

zeros_likeConst*
valueBђ*    *
dtype0ё
concat/ReadVariableOpReadVariableOpbias^zeros_like/ReadVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
dtype05
concat/axisConst*
value	B : *
dtype0n
concatConcatV2zeros_like:output:0concat/ReadVariableOp:value:0concat/axis:output:0*
T0*
N1
Const_2Const*
dtype0*
value	B :;
split_2/split_dimConst*
dtype0*
value	B : W
split_2Splitsplit_2/split_dim:output:0concat:output:0*
T0*
	num_split>
Const_3Const*
valueB:
         *
dtype0E
transpose_1/permConst*
valueB"       *
dtype0L
transpose_1	Transposesplit:output:0transpose_1/perm:output:0*
T0>
ReshapeReshapetranspose_1:y:0Const_3:output:0*
T0E
transpose_2/permConst*
valueB"       *
dtype0L
transpose_2	Transposesplit:output:1transpose_2/perm:output:0*
T0@
	Reshape_1Reshapetranspose_2:y:0Const_3:output:0*
T0E
transpose_3/permConst*
valueB"       *
dtype0L
transpose_3	Transposesplit:output:2transpose_3/perm:output:0*
T0@
	Reshape_2Reshapetranspose_3:y:0Const_3:output:0*
T0E
transpose_4/permConst*
dtype0*
valueB"       L
transpose_4	Transposesplit:output:3transpose_4/perm:output:0*
T0@
	Reshape_3Reshapetranspose_4:y:0Const_3:output:0*
T0E
transpose_5/permConst*
valueB"       *
dtype0N
transpose_5	Transposesplit_1:output:0transpose_5/perm:output:0*
T0@
	Reshape_4Reshapetranspose_5:y:0Const_3:output:0*
T0E
transpose_6/permConst*
dtype0*
valueB"       N
transpose_6	Transposesplit_1:output:1transpose_6/perm:output:0*
T0@
	Reshape_5Reshapetranspose_6:y:0Const_3:output:0*
T0E
transpose_7/permConst*
dtype0*
valueB"       N
transpose_7	Transposesplit_1:output:2transpose_7/perm:output:0*
T0@
	Reshape_6Reshapetranspose_7:y:0Const_3:output:0*
T0E
transpose_8/permConst*
valueB"       *
dtype0N
transpose_8	Transposesplit_1:output:3transpose_8/perm:output:0*
T0@
	Reshape_7Reshapetranspose_8:y:0Const_3:output:0*
T0A
	Reshape_8Reshapesplit_2:output:0Const_3:output:0*
T0A
	Reshape_9Reshapesplit_2:output:1Const_3:output:0*
T0B

Reshape_10Reshapesplit_2:output:2Const_3:output:0*
T0B

Reshape_11Reshapesplit_2:output:3Const_3:output:0*
T0B

Reshape_12Reshapesplit_2:output:4Const_3:output:0*
T0B

Reshape_13Reshapesplit_2:output:5Const_3:output:0*
T0B

Reshape_14Reshapesplit_2:output:6Const_3:output:0*
T0B

Reshape_15Reshapesplit_2:output:7Const_3:output:0*
T07
concat_1/axisConst*
value	B : *
dtype0ё

concat_1_0ConcatV2Reshape:output:0Reshape_1:output:0Reshape_2:output:0Reshape_3:output:0Reshape_4:output:0Reshape_5:output:0Reshape_6:output:0Reshape_7:output:0Reshape_8:output:0Reshape_9:output:0Reshape_10:output:0Reshape_11:output:0Reshape_12:output:0Reshape_13:output:0Reshape_14:output:0Reshape_15:output:0concat_1/axis:output:0*
N*
T0o
CudnnRNNCudnnRNNtranspose_0:y:0ExpandDims:output:0ExpandDims_1:output:0concat_1_0:output:0*
T0J
strided_slice/stackConst*
valueB:
         *
dtype0C
strided_slice/stack_1Const*
dtype0*
valueB: C
strided_slice/stack_2Const*
valueB:*
dtype0╝
strided_sliceStridedSliceCudnnRNN:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_maskI
transpose_9/permConst*!
valueB"          *
dtype0O
transpose_9	TransposeCudnnRNN:output:0transpose_9/perm:output:0*
T0C
strided_slice_1/stackConst*
valueB: *
dtype0E
strided_slice_1/stack_1Const*
dtype0*
valueB:E
strided_slice_1/stack_2Const*
valueB:*
dtype0к
strided_slice_1StridedSliceCudnnRNN:output_h:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
T0*
Index0*
shrink_axis_maskC
strided_slice_2/stackConst*
valueB: *
dtype0E
strided_slice_2/stack_1Const*
valueB:*
dtype0E
strided_slice_2/stack_2Const*
valueB:*
dtype0к
strided_slice_2StridedSliceCudnnRNN:output_c:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
shrink_axis_mask*
T0*
Index0C
runtimeConst"/device:CPU:0*
dtype0*
valueB
 *   @ц
IdentityIdentitystrided_slice:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0Ъ

Identity_1Identitytranspose_9:y:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_2Identitystrided_slice_1:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0е

Identity_3Identitystrided_slice_2:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0а

Identity_4Identityruntime:output:0	^CudnnRNN^concat/ReadVariableOp^split/ReadVariableOp^split_1/ReadVariableOp^zeros_like/ReadVariableOp*
T0"
init_cinit_c_0";
strided_slice_2_stack_1 strided_slice_2/stack_1:output:0";
strided_slice_2_stack_2 strided_slice_2/stack_2:output:0"7
strided_slice_stack_1strided_slice/stack_1:output:0"-
transpose_6_permtranspose_6/perm:output:0"
init_hinit_h_0"7
strided_slice_stack_2strided_slice/stack_2:output:0"!

cudnnrnn_0CudnnRNN:output_h:0"!

cudnnrnn_1CudnnRNN:output_c:0"&

cudnnrnn_2CudnnRNN:reserve_space:0"-
transpose_5_permtranspose_5/perm:output:0"
cudnnrnnCudnnRNN:output:0")
transpose_permtranspose/perm:output:0"!

expanddimsExpandDims:output:0"-
transpose_4_permtranspose_4/perm:output:0"%
expanddims_1ExpandDims_1:output:0"/
split_2_split_dimsplit_2/split_dim:output:0";
strided_slice_1_stack_1 strided_slice_1/stack_1:output:0"
concat_1concat_1_0:output:0";
strided_slice_1_stack_2 strided_slice_1/stack_2:output:0"-
transpose_9_permtranspose_9/perm:output:0"-
transpose_3_permtranspose_3/perm:output:0"/
split_1_split_dimsplit_1/split_dim:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"7
strided_slice_1_stackstrided_slice_1/stack:output:0"!

identity_3Identity_3:output:0"-
transpose_8_permtranspose_8/perm:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"3
strided_slice_stackstrided_slice/stack:output:0"-
transpose_2_permtranspose_2/perm:output:0"7
strided_slice_2_stackstrided_slice_2/stack:output:0"'
concat_1_axisconcat_1/axis:output:0"
	transposetranspose_0:y:0"#
concat_axisconcat/axis:output:0"+
split_split_dimsplit/split_dim:output:0"-
transpose_7_permtranspose_7/perm:output:0"-
transpose_1_permtranspose_1/perm:output:0*W
backward_function_name=;__inference___backward_cudnn_lstm_with_fallback_12932_13114*
_input_shapes *=
api_implements+)lstm_fa5d5f1a-3419-4659-95e4-7ff9aa932079*
api_preferred_deviceGPU2
CudnnRNNCudnnRNN2.
concat/ReadVariableOpconcat/ReadVariableOp2,
split/ReadVariableOpsplit/ReadVariableOp26
zeros_like/ReadVariableOpzeros_like/ReadVariableOp20
split_1/ReadVariableOpsplit_1/ReadVariableOp:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h
¤+
Ь
while_body_12717
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"       ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Е
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђe
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
dtype0*
_output_shapes
: *
value	B :Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*
T0*'
_output_shapes
:         @T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
_output_shapes
: *
T0I
add_3/yConst*
dtype0*
_output_shapes
: *
value	B :U
add_3AddV2while_loop_counteradd_3/y:output:0*
_output_shapes
: *
T0І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
_output_shapes
: *
T0ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0*Q
_input_shapes@
>: : : : :         @:         @: : :::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp:  : : : : : : : : :	 :
 
п
љ
while_cond_12716
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : :	 :
 :  : : : : : 
█
Љ
while_cond_9770
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice_1+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
P
LessLessplaceholderless_strided_slice_1*
_output_shapes
: *
T0?
IdentityIdentityLess:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::: : : : : : : :	 :
 :  : 
ыG
Ћ
__inference_standard_lstm_10040

inputs

init_h

init_c

kernel
recurrent_kernel
bias
identity

identity_1

identity_2

identity_3

identity_4ѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpбwhilec
transpose/permConst*!
valueB"          *
dtype0*
_output_shapes
:m
	transpose	Transposeinputstranspose/perm:output:0*
T0*+
_output_shapes
:Z         B
ShapeShapetranspose:y:0*
T0*
_output_shapes
:]
strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: _
strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
shrink_axis_mask*
_output_shapes
: *
T0*
Index0f
TensorArrayV2/element_shapeConst*
dtype0*
_output_shapes
: *
valueB :
         Ъ
TensorArrayV2TensorListReserve$TensorArrayV2/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: є
5TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
valueB"       *
dtype0*
_output_shapes
:═
'TensorArrayUnstack/TensorListFromTensorTensorListFromTensortranspose:y:0>TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*

shape_type0*
element_dtype0*
_output_shapes
: _
strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:a
strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:ж
strided_slice_1StridedSlicetranspose:y:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*'
_output_shapes
:         *
T0*
Index0*
shrink_axis_maskІ
MatMul/ReadVariableOpReadVariableOpkernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђ|
MatMulMatMulstrided_slice_1:output:0MatMul/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0Ќ
MatMul_1/ReadVariableOpReadVariableOprecurrent_kernel",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђn
MatMul_1MatMulinit_hMatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђє
BiasAdd/ReadVariableOpReadVariableOpbias",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0G
ConstConst*
dtype0*
_output_shapes
: *
value	B :Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*
T0*'
_output_shapes
:         @V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @S
mulMulSigmoid_1:y:0init_c*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*'
_output_shapes
:         @*
T0U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*'
_output_shapes
:         @*
T0V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*
T0*'
_output_shapes
:         @n
TensorArrayV2_1/element_shapeConst*
valueB"    @   *
dtype0*
_output_shapes
:Б
TensorArrayV2_1TensorListReserve&TensorArrayV2_1/element_shape:output:0strided_slice:output:0*

shape_type0*
element_dtype0*
_output_shapes
: F
timeConst*
value	B : *
dtype0*
_output_shapes
: Z
while/maximum_iterationsConst*
dtype0*
_output_shapes
: *
value	B :ZT
while/loop_counterConst*
value	B : *
dtype0*
_output_shapes
: Є
whileWhilewhile/loop_counter:output:0!while/maximum_iterations:output:0time:output:0TensorArrayV2_1:handle:0init_hinit_cstrided_slice:output:07TensorArrayUnstack/TensorListFromTensor:output_handle:0kernelrecurrent_kernelbias^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
parallel_iterations *
condR
while_cond_9934*
_num_original_outputs*
bodyR
while_body_9935*L
_output_shapes:
8: : : : :         @:         @: : : : : *
T
2*K
output_shapes:
8: : : : :         @:         @: : : : : *
_lower_using_switch_merge(K
while/IdentityIdentitywhile:output:0*
T0*
_output_shapes
: M
while/Identity_1Identitywhile:output:1*
T0*
_output_shapes
: M
while/Identity_2Identitywhile:output:2*
T0*
_output_shapes
: M
while/Identity_3Identitywhile:output:3*
T0*
_output_shapes
: ^
while/Identity_4Identitywhile:output:4*
T0*'
_output_shapes
:         @^
while/Identity_5Identitywhile:output:5*
T0*'
_output_shapes
:         @M
while/Identity_6Identitywhile:output:6*
T0*
_output_shapes
: M
while/Identity_7Identitywhile:output:7*
T0*
_output_shapes
: M
while/Identity_8Identitywhile:output:8*
T0*
_output_shapes
: M
while/Identity_9Identitywhile:output:9*
T0*
_output_shapes
: O
while/Identity_10Identitywhile:output:10*
_output_shapes
: *
T0Ђ
0TensorArrayV2Stack/TensorListStack/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   ═
"TensorArrayV2Stack/TensorListStackTensorListStackwhile/Identity_3:output:09TensorArrayV2Stack/TensorListStack/element_shape:output:0*
element_dtype0*+
_output_shapes
:Z         @h
strided_slice_2/stackConst*
valueB:
         *
dtype0*
_output_shapes
:a
strided_slice_2/stack_1Const*
valueB: *
dtype0*
_output_shapes
:a
strided_slice_2/stack_2Const*
valueB:*
dtype0*
_output_shapes
:Є
strided_slice_2StridedSlice+TensorArrayV2Stack/TensorListStack:tensor:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*'
_output_shapes
:         @e
transpose_1/permConst*
dtype0*
_output_shapes
:*!
valueB"          ќ
transpose_1	Transpose+TensorArrayV2Stack/TensorListStack:tensor:0transpose_1/perm:output:0*
T0*+
_output_shapes
:         Z@[
runtimeConst"/device:CPU:0*
valueB
 *  ђ?*
dtype0*
_output_shapes
: │
IdentityIdentitystrided_slice_2:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @░

Identity_1Identitytranspose_1:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*+
_output_shapes
:         Z@*
T0Х

Identity_2Identitywhile/Identity_4:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
T0*'
_output_shapes
:         @Х

Identity_3Identitywhile/Identity_5:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*'
_output_shapes
:         @*
T0ю

Identity_4Identityruntime:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp^while*
_output_shapes
: *
T0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0*\
_input_shapesK
I:         Z:         @:         @:::*=
api_implements+)lstm_4e96b1c4-3fa5-495d-9170-91a62470ff34*
api_preferred_deviceCPU22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp2
whilewhile2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinit_h:&"
 
_user_specified_nameinit_c:&"
 
_user_specified_namekernel:0,
*
_user_specified_namerecurrent_kernel:$ 

_user_specified_namebias
▓
с
>__inference_lstm_layer_call_and_return_conditional_losses_8727

inputs"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5
identityѕбStatefulPartitionedCall;
ShapeShapeinputs*
T0*
_output_shapes
:]
strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:_
strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:_
strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:Л
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
shrink_axis_mask*
_output_shapes
: M
zeros/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: _
	zeros/mulMulstrided_slice:output:0zeros/mul/y:output:0*
T0*
_output_shapes
: O
zeros/Less/yConst*
value
B :У*
dtype0*
_output_shapes
: Y

zeros/LessLesszeros/mul:z:0zeros/Less/y:output:0*
T0*
_output_shapes
: P
zeros/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: s
zeros/packedPackstrided_slice:output:0zeros/packed/1:output:0*
T0*
N*
_output_shapes
:P
zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    l
zerosFillzeros/packed:output:0zeros/Const:output:0*
T0*'
_output_shapes
:         @O
zeros_1/mul/yConst*
value	B :@*
dtype0*
_output_shapes
: c
zeros_1/mulMulstrided_slice:output:0zeros_1/mul/y:output:0*
_output_shapes
: *
T0Q
zeros_1/Less/yConst*
dtype0*
_output_shapes
: *
value
B :У_
zeros_1/LessLesszeros_1/mul:z:0zeros_1/Less/y:output:0*
T0*
_output_shapes
: R
zeros_1/packed/1Const*
value	B :@*
dtype0*
_output_shapes
: w
zeros_1/packedPackstrided_slice:output:0zeros_1/packed/1:output:0*
T0*
N*
_output_shapes
:R
zeros_1/ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: r
zeros_1Fillzeros_1/packed:output:0zeros_1/Const:output:0*
T0*'
_output_shapes
:         @Н
StatefulPartitionedCallStatefulPartitionedCallinputszeros:output:0zeros_1:output:0statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5*
Tin

2*o
_output_shapes]
[:         @:                  @:         @:         @: *+
_gradient_op_typePartitionedCall-8432*'
f"R 
__inference_standard_lstm_8431*
Tout	
2**
config_proto

GPU 

CPU2J 8Ј
IdentityIdentity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*?
_input_shapes.
,:                  :::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : : 
Э─
¤
;__inference___backward_cudnn_lstm_with_fallback_13892_14074
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*
T0*'
_output_shapes
:         @d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         Z@`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:┤
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:Z         @ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:и
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:Z         @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
shrink_axis_mask*+
_output_shapes
:         @*
T0*
Index0Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*+
_output_shapes
:         @*
Index0*
T0*
shrink_axis_maskщ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:Z         @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
_output_shapes
:*
T0»
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:Z         :         @:         @:ђїќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:         Zu
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
T0*
_output_shapes
:┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_2Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_3Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_4Const*
dtype0*
_output_shapes
:*
valueB:ђ j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
_output_shapes	
:ђ*
Index0*
T0­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
_output_shapes	
:ђ *
Index0*
T0№
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
_output_shapes
:@*
Index0*
T0№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_5_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_10_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
T0*
N*
_output_shapes	
:ђ»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	@ђ\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђ╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*+
_output_shapes
:         Z*
T0Ъ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	@ђЈ

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_14073*Ъ
_input_shapesЇ
і:         @:         Z@:         @:         @: :Z         @:::::         @::::         @:::::Z         :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_af332698-df81-42b0-b14c-2c32c2159570*
api_preferred_deviceGPU2T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
ё
Ј
while_cond_6733
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
less_strided_slice+
'tensorarrayunstack_tensorlistfromtensor
unknown
	unknown_0
	unknown_1
identity
N
LessLessplaceholderless_strided_slice*
T0*
_output_shapes
: ]
Less_1Lesswhile_loop_counterwhile_maximum_iterations*
T0*
_output_shapes
: F

LogicalAnd
LogicalAnd
Less_1:z:0Less:z:0*
_output_shapes
: E
IdentityIdentityLogicalAnd:z:0*
T0
*
_output_shapes
: "
identityIdentity:output:0*Q
_input_shapes@
>: : : : :         @:         @: : ::::  : : : : : : : : :	 :
 
¤
д
%__inference_dense_layer_call_fn_15260

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identityѕбStatefulPartitionedCallу
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*,
_gradient_op_typePartitionedCall-11204*I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_11203*
Tout
2**
config_proto

GPU 

CPU2J 8*
Tin
2*'
_output_shapes
:         ѓ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:         "
identityIdentity:output:0*.
_input_shapes
:         @::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs: : 
Э─
¤
;__inference___backward_cudnn_lstm_with_fallback_12248_12430
placeholder
placeholder_1
placeholder_2
placeholder_3
placeholder_4/
+gradients_strided_slice_grad_shape_cudnnrnnE
Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackG
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1G
Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2A
=gradients_transpose_9_grad_invertpermutation_transpose_9_perm1
-gradients_strided_slice_1_grad_shape_cudnnrnnI
Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackK
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1K
Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_21
-gradients_strided_slice_2_grad_shape_cudnnrnnI
Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackK
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1K
Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2!
gradients_zeros_like_cudnnrnn6
2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose7
3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims9
5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_15
1gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1=
9gradients_transpose_grad_invertpermutation_transpose_perm*
&gradients_expanddims_grad_shape_init_h,
(gradients_expanddims_1_grad_shape_init_c-
)gradients_concat_1_grad_mod_concat_1_axisA
=gradients_transpose_1_grad_invertpermutation_transpose_1_permA
=gradients_transpose_2_grad_invertpermutation_transpose_2_permA
=gradients_transpose_3_grad_invertpermutation_transpose_3_permA
=gradients_transpose_4_grad_invertpermutation_transpose_4_permA
=gradients_transpose_5_grad_invertpermutation_transpose_5_permA
=gradients_transpose_6_grad_invertpermutation_transpose_6_permA
=gradients_transpose_7_grad_invertpermutation_transpose_7_permA
=gradients_transpose_8_grad_invertpermutation_transpose_8_perm3
/gradients_split_2_grad_concat_split_2_split_dim/
+gradients_split_grad_concat_split_split_dim3
/gradients_split_1_grad_concat_split_1_split_dim)
%gradients_concat_grad_mod_concat_axis
identity

identity_1

identity_2

identity_3

identity_4

identity_5ѕб(gradients/CudnnRNN_grad/CudnnRNNBackprop^
gradients/grad_ys_0Identityplaceholder*'
_output_shapes
:         @*
T0d
gradients/grad_ys_1Identityplaceholder_1*
T0*+
_output_shapes
:         Z@`
gradients/grad_ys_2Identityplaceholder_2*
T0*'
_output_shapes
:         @`
gradients/grad_ys_3Identityplaceholder_3*
T0*'
_output_shapes
:         @O
gradients/grad_ys_4Identityplaceholder_4*
_output_shapes
: *
T0}
"gradients/strided_slice_grad/ShapeShape+gradients_strided_slice_grad_shape_cudnnrnn*
T0*
_output_shapes
:┤
-gradients/strided_slice_grad/StridedSliceGradStridedSliceGrad+gradients/strided_slice_grad/Shape:output:0Agradients_strided_slice_grad_stridedslicegrad_strided_slice_stackCgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_1Cgradients_strided_slice_grad_stridedslicegrad_strided_slice_stack_2gradients/grad_ys_0:output:0*
Index0*
T0*
shrink_axis_mask*+
_output_shapes
:Z         @ю
,gradients/transpose_9_grad/InvertPermutationInvertPermutation=gradients_transpose_9_grad_invertpermutation_transpose_9_perm*
_output_shapes
:и
$gradients/transpose_9_grad/transpose	Transposegradients/grad_ys_1:output:00gradients/transpose_9_grad/InvertPermutation:y:0*
T0*+
_output_shapes
:Z         @Ђ
$gradients/strided_slice_1_grad/ShapeShape-gradients_strided_slice_1_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_1_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_1_grad/Shape:output:0Egradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stackGgradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_1Ggradients_strided_slice_1_grad_stridedslicegrad_strided_slice_1_stack_2gradients/grad_ys_2:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:         @Ђ
$gradients/strided_slice_2_grad/ShapeShape-gradients_strided_slice_2_grad_shape_cudnnrnn*
T0*
_output_shapes
:─
/gradients/strided_slice_2_grad/StridedSliceGradStridedSliceGrad-gradients/strided_slice_2_grad/Shape:output:0Egradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stackGgradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_1Ggradients_strided_slice_2_grad_stridedslicegrad_strided_slice_2_stack_2gradients/grad_ys_3:output:0*
T0*
Index0*
shrink_axis_mask*+
_output_shapes
:         @щ
gradients/AddNAddN6gradients/strided_slice_grad/StridedSliceGrad:output:0(gradients/transpose_9_grad/transpose:y:0*
T0*@
_class6
42loc:@gradients/strided_slice_grad/StridedSliceGrad*
N*+
_output_shapes
:Z         @c
gradients/zeros_like	ZerosLikegradients_zeros_like_cudnnrnn*
T0*
_output_shapes
:»
(gradients/CudnnRNN_grad/CudnnRNNBackpropCudnnRNNBackprop2gradients_cudnnrnn_grad_cudnnrnnbackprop_transpose3gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims5gradients_cudnnrnn_grad_cudnnrnnbackprop_expanddims_11gradients_cudnnrnn_grad_cudnnrnnbackprop_concat_1+gradients_strided_slice_grad_shape_cudnnrnn-gradients_strided_slice_1_grad_shape_cudnnrnn-gradients_strided_slice_2_grad_shape_cudnnrnngradients/AddN:sum:08gradients/strided_slice_1_grad/StridedSliceGrad:output:08gradients/strided_slice_2_grad/StridedSliceGrad:output:0gradients_zeros_like_cudnnrnn*
T0*a
_output_shapesO
M:Z         :         @:         @:ђїќ
*gradients/transpose_grad/InvertPermutationInvertPermutation9gradients_transpose_grad_invertpermutation_transpose_perm*
_output_shapes
:л
"gradients/transpose_grad/transpose	Transpose9gradients/CudnnRNN_grad/CudnnRNNBackprop:input_backprop:0.gradients/transpose_grad/InvertPermutation:y:0*+
_output_shapes
:         Z*
T0u
gradients/ExpandDims_grad/ShapeShape&gradients_expanddims_grad_shape_init_h*
_output_shapes
:*
T0┼
!gradients/ExpandDims_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_h_backprop:0(gradients/ExpandDims_grad/Shape:output:0*
T0*'
_output_shapes
:         @y
!gradients/ExpandDims_1_grad/ShapeShape(gradients_expanddims_1_grad_shape_init_c*
T0*
_output_shapes
:╔
#gradients/ExpandDims_1_grad/ReshapeReshape;gradients/CudnnRNN_grad/CudnnRNNBackprop:input_c_backprop:0*gradients/ExpandDims_1_grad/Shape:output:0*
T0*'
_output_shapes
:         @^
gradients/concat_1_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: џ
gradients/concat_1_grad/modFloorMod)gradients_concat_1_grad_mod_concat_1_axis%gradients/concat_1_grad/Rank:output:0*
T0*
_output_shapes
: h
gradients/concat_1_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_1Const*
valueB:ђ*
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_2Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_3Const*
dtype0*
_output_shapes
:*
valueB:ђj
gradients/concat_1_grad/Shape_4Const*
dtype0*
_output_shapes
:*
valueB:ђ j
gradients/concat_1_grad/Shape_5Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_6Const*
valueB:ђ *
dtype0*
_output_shapes
:j
gradients/concat_1_grad/Shape_7Const*
valueB:ђ *
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_8Const*
valueB:@*
dtype0*
_output_shapes
:i
gradients/concat_1_grad/Shape_9Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_10Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_11Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_12Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_13Const*
dtype0*
_output_shapes
:*
valueB:@j
 gradients/concat_1_grad/Shape_14Const*
valueB:@*
dtype0*
_output_shapes
:j
 gradients/concat_1_grad/Shape_15Const*
valueB:@*
dtype0*
_output_shapes
:Э
$gradients/concat_1_grad/ConcatOffsetConcatOffsetgradients/concat_1_grad/mod:z:0&gradients/concat_1_grad/Shape:output:0(gradients/concat_1_grad/Shape_1:output:0(gradients/concat_1_grad/Shape_2:output:0(gradients/concat_1_grad/Shape_3:output:0(gradients/concat_1_grad/Shape_4:output:0(gradients/concat_1_grad/Shape_5:output:0(gradients/concat_1_grad/Shape_6:output:0(gradients/concat_1_grad/Shape_7:output:0(gradients/concat_1_grad/Shape_8:output:0(gradients/concat_1_grad/Shape_9:output:0)gradients/concat_1_grad/Shape_10:output:0)gradients/concat_1_grad/Shape_11:output:0)gradients/concat_1_grad/Shape_12:output:0)gradients/concat_1_grad/Shape_13:output:0)gradients/concat_1_grad/Shape_14:output:0)gradients/concat_1_grad/Shape_15:output:0*
N*t
_output_shapesb
`::::::::::::::::В
gradients/concat_1_grad/SliceSlice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:0&gradients/concat_1_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_1Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:1(gradients/concat_1_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_2Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:2(gradients/concat_1_grad/Shape_2:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_3Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:3(gradients/concat_1_grad/Shape_3:output:0*
Index0*
T0*
_output_shapes	
:ђ­
gradients/concat_1_grad/Slice_4Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:4(gradients/concat_1_grad/Shape_4:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_5Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:5(gradients/concat_1_grad/Shape_5:output:0*
Index0*
T0*
_output_shapes	
:ђ ­
gradients/concat_1_grad/Slice_6Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:6(gradients/concat_1_grad/Shape_6:output:0*
_output_shapes	
:ђ *
Index0*
T0­
gradients/concat_1_grad/Slice_7Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:7(gradients/concat_1_grad/Shape_7:output:0*
Index0*
T0*
_output_shapes	
:ђ №
gradients/concat_1_grad/Slice_8Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:8(gradients/concat_1_grad/Shape_8:output:0*
Index0*
T0*
_output_shapes
:@№
gradients/concat_1_grad/Slice_9Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0-gradients/concat_1_grad/ConcatOffset:offset:9(gradients/concat_1_grad/Shape_9:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_10Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:10)gradients/concat_1_grad/Shape_10:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_11Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:11)gradients/concat_1_grad/Shape_11:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_12Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:12)gradients/concat_1_grad/Shape_12:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_13Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:13)gradients/concat_1_grad/Shape_13:output:0*
Index0*
T0*
_output_shapes
:@Ы
 gradients/concat_1_grad/Slice_14Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:14)gradients/concat_1_grad/Shape_14:output:0*
_output_shapes
:@*
Index0*
T0Ы
 gradients/concat_1_grad/Slice_15Slice:gradients/CudnnRNN_grad/CudnnRNNBackprop:params_backprop:0.gradients/concat_1_grad/ConcatOffset:offset:15)gradients/concat_1_grad/Shape_15:output:0*
Index0*
T0*
_output_shapes
:@m
gradients/Reshape_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:А
gradients/Reshape_grad/ReshapeReshape&gradients/concat_1_grad/Slice:output:0%gradients/Reshape_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_1_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_1_grad/ReshapeReshape(gradients/concat_1_grad/Slice_1:output:0'gradients/Reshape_1_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_2_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_2_grad/ReshapeReshape(gradients/concat_1_grad/Slice_2:output:0'gradients/Reshape_2_grad/Shape:output:0*
T0*
_output_shapes

:@o
gradients/Reshape_3_grad/ShapeConst*
valueB"@      *
dtype0*
_output_shapes
:Д
 gradients/Reshape_3_grad/ReshapeReshape(gradients/concat_1_grad/Slice_3:output:0'gradients/Reshape_3_grad/Shape:output:0*
_output_shapes

:@*
T0o
gradients/Reshape_4_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_4_grad/ReshapeReshape(gradients/concat_1_grad/Slice_4:output:0'gradients/Reshape_4_grad/Shape:output:0*
_output_shapes

:@@*
T0o
gradients/Reshape_5_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB"@   @   Д
 gradients/Reshape_5_grad/ReshapeReshape(gradients/concat_1_grad/Slice_5:output:0'gradients/Reshape_5_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_6_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_6_grad/ReshapeReshape(gradients/concat_1_grad/Slice_6:output:0'gradients/Reshape_6_grad/Shape:output:0*
T0*
_output_shapes

:@@o
gradients/Reshape_7_grad/ShapeConst*
valueB"@   @   *
dtype0*
_output_shapes
:Д
 gradients/Reshape_7_grad/ReshapeReshape(gradients/concat_1_grad/Slice_7:output:0'gradients/Reshape_7_grad/Shape:output:0*
T0*
_output_shapes

:@@h
gradients/Reshape_8_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@Б
 gradients/Reshape_8_grad/ReshapeReshape(gradients/concat_1_grad/Slice_8:output:0'gradients/Reshape_8_grad/Shape:output:0*
T0*
_output_shapes
:@h
gradients/Reshape_9_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:Б
 gradients/Reshape_9_grad/ReshapeReshape(gradients/concat_1_grad/Slice_9:output:0'gradients/Reshape_9_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_10_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_10_grad/ReshapeReshape)gradients/concat_1_grad/Slice_10:output:0(gradients/Reshape_10_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_11_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_11_grad/ReshapeReshape)gradients/concat_1_grad/Slice_11:output:0(gradients/Reshape_11_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_12_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_12_grad/ReshapeReshape)gradients/concat_1_grad/Slice_12:output:0(gradients/Reshape_12_grad/Shape:output:0*
_output_shapes
:@*
T0i
gradients/Reshape_13_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_13_grad/ReshapeReshape)gradients/concat_1_grad/Slice_13:output:0(gradients/Reshape_13_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_14_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:@д
!gradients/Reshape_14_grad/ReshapeReshape)gradients/concat_1_grad/Slice_14:output:0(gradients/Reshape_14_grad/Shape:output:0*
T0*
_output_shapes
:@i
gradients/Reshape_15_grad/ShapeConst*
valueB:@*
dtype0*
_output_shapes
:д
!gradients/Reshape_15_grad/ReshapeReshape)gradients/concat_1_grad/Slice_15:output:0(gradients/Reshape_15_grad/Shape:output:0*
T0*
_output_shapes
:@ю
,gradients/transpose_1_grad/InvertPermutationInvertPermutation=gradients_transpose_1_grad_invertpermutation_transpose_1_perm*
_output_shapes
:х
$gradients/transpose_1_grad/transpose	Transpose'gradients/Reshape_grad/Reshape:output:00gradients/transpose_1_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_2_grad/InvertPermutationInvertPermutation=gradients_transpose_2_grad_invertpermutation_transpose_2_perm*
_output_shapes
:и
$gradients/transpose_2_grad/transpose	Transpose)gradients/Reshape_1_grad/Reshape:output:00gradients/transpose_2_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_3_grad/InvertPermutationInvertPermutation=gradients_transpose_3_grad_invertpermutation_transpose_3_perm*
_output_shapes
:и
$gradients/transpose_3_grad/transpose	Transpose)gradients/Reshape_2_grad/Reshape:output:00gradients/transpose_3_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@ю
,gradients/transpose_4_grad/InvertPermutationInvertPermutation=gradients_transpose_4_grad_invertpermutation_transpose_4_perm*
_output_shapes
:и
$gradients/transpose_4_grad/transpose	Transpose)gradients/Reshape_3_grad/Reshape:output:00gradients/transpose_4_grad/InvertPermutation:y:0*
_output_shapes

:@*
T0ю
,gradients/transpose_5_grad/InvertPermutationInvertPermutation=gradients_transpose_5_grad_invertpermutation_transpose_5_perm*
_output_shapes
:и
$gradients/transpose_5_grad/transpose	Transpose)gradients/Reshape_4_grad/Reshape:output:00gradients/transpose_5_grad/InvertPermutation:y:0*
T0*
_output_shapes

:@@ю
,gradients/transpose_6_grad/InvertPermutationInvertPermutation=gradients_transpose_6_grad_invertpermutation_transpose_6_perm*
_output_shapes
:и
$gradients/transpose_6_grad/transpose	Transpose)gradients/Reshape_5_grad/Reshape:output:00gradients/transpose_6_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_7_grad/InvertPermutationInvertPermutation=gradients_transpose_7_grad_invertpermutation_transpose_7_perm*
_output_shapes
:и
$gradients/transpose_7_grad/transpose	Transpose)gradients/Reshape_6_grad/Reshape:output:00gradients/transpose_7_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0ю
,gradients/transpose_8_grad/InvertPermutationInvertPermutation=gradients_transpose_8_grad_invertpermutation_transpose_8_perm*
_output_shapes
:и
$gradients/transpose_8_grad/transpose	Transpose)gradients/Reshape_7_grad/Reshape:output:00gradients/transpose_8_grad/InvertPermutation:y:0*
_output_shapes

:@@*
T0у
gradients/split_2_grad/concatConcatV2)gradients/Reshape_8_grad/Reshape:output:0)gradients/Reshape_9_grad/Reshape:output:0*gradients/Reshape_10_grad/Reshape:output:0*gradients/Reshape_11_grad/Reshape:output:0*gradients/Reshape_12_grad/Reshape:output:0*gradients/Reshape_13_grad/Reshape:output:0*gradients/Reshape_14_grad/Reshape:output:0*gradients/Reshape_15_grad/Reshape:output:0/gradients_split_2_grad_concat_split_2_split_dim*
N*
_output_shapes	
:ђ*
T0»
gradients/split_grad/concatConcatV2(gradients/transpose_1_grad/transpose:y:0(gradients/transpose_2_grad/transpose:y:0(gradients/transpose_3_grad/transpose:y:0(gradients/transpose_4_grad/transpose:y:0+gradients_split_grad_concat_split_split_dim*
T0*
N*
_output_shapes
:	ђх
gradients/split_1_grad/concatConcatV2(gradients/transpose_5_grad/transpose:y:0(gradients/transpose_6_grad/transpose:y:0(gradients/transpose_7_grad/transpose:y:0(gradients/transpose_8_grad/transpose:y:0/gradients_split_1_grad_concat_split_1_split_dim*
T0*
N*
_output_shapes
:	@ђ\
gradients/concat_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: њ
gradients/concat_grad/modFloorMod%gradients_concat_grad_mod_concat_axis#gradients/concat_grad/Rank:output:0*
T0*
_output_shapes
: f
gradients/concat_grad/ShapeConst*
valueB:ђ*
dtype0*
_output_shapes
:h
gradients/concat_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:ђ╩
"gradients/concat_grad/ConcatOffsetConcatOffsetgradients/concat_grad/mod:z:0$gradients/concat_grad/Shape:output:0&gradients/concat_grad/Shape_1:output:0*
N* 
_output_shapes
::м
gradients/concat_grad/SliceSlice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:0$gradients/concat_grad/Shape:output:0*
Index0*
T0*
_output_shapes	
:ђо
gradients/concat_grad/Slice_1Slice&gradients/split_2_grad/concat:output:0+gradients/concat_grad/ConcatOffset:offset:1&gradients/concat_grad/Shape_1:output:0*
Index0*
T0*
_output_shapes	
:ђЮ
IdentityIdentity&gradients/transpose_grad/transpose:y:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*+
_output_shapes
:         ZЪ

Identity_1Identity*gradients/ExpandDims_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @А

Identity_2Identity,gradients/ExpandDims_1_grad/Reshape:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*'
_output_shapes
:         @Љ

Identity_3Identity$gradients/split_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes
:	ђЊ

Identity_4Identity&gradients/split_1_grad/concat:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
_output_shapes
:	@ђ*
T0Ј

Identity_5Identity&gradients/concat_grad/Slice_1:output:0)^gradients/CudnnRNN_grad/CudnnRNNBackprop*
T0*
_output_shapes	
:ђ"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0"
identityIdentity:output:0"!

identity_5Identity_5:output:0*Ъ
_input_shapesЇ
і:         @:         Z@:         @:         @: :Z         @:::::         @::::         @:::::Z         :         @:         @:ђї::         @:         @: ::::::::: : : : *=
api_implements+)lstm_5e66a3d3-3a72-45c8-be76-304c5e537995*
api_preferred_deviceGPU*C
forward_function_name*(__forward_cudnn_lstm_with_fallback_124292T
(gradients/CudnnRNN_grad/CudnnRNNBackprop(gradients/CudnnRNN_grad/CudnnRNNBackprop:  : : : : : : : : :	 :
 : : : : : : : : : : : : : : : : : : : : : :  :! :" :# :$ :% :& 
╬+
ь
while_body_8328
while_loop_counter
while_maximum_iterations
placeholder
placeholder_1
placeholder_2
placeholder_3
strided_slice_0Q
Mtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0$
 matmul_readvariableop_resource_0&
"matmul_1_readvariableop_resource_0%
!biasadd_readvariableop_resource_0
identity

identity_1

identity_2

identity_3

identity_4

identity_5
strided_sliceO
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor"
matmul_readvariableop_resource$
 matmul_1_readvariableop_resource#
biasadd_readvariableop_resourceѕбBiasAdd/ReadVariableOpбMatMul/ReadVariableOpбMatMul_1/ReadVariableOpѓ
1TensorArrayV2Read/TensorListGetItem/element_shapeConst*
dtype0*
_output_shapes
:*
valueB"       ј
#TensorArrayV2Read/TensorListGetItemTensorListGetItemMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0placeholder:TensorArrayV2Read/TensorListGetItem/element_shape:output:0*
element_dtype0*'
_output_shapes
:         Ц
MatMul/ReadVariableOpReadVariableOp matmul_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	ђј
MatMulMatMul*TensorArrayV2Read/TensorListGetItem:item:0MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђЕ
MatMul_1/ReadVariableOpReadVariableOp"matmul_1_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes
:	@ђu
MatMul_1MatMulplaceholder_2MatMul_1/ReadVariableOp:value:0*(
_output_shapes
:         ђ*
T0e
addAddV2MatMul:product:0MatMul_1:product:0*
T0*(
_output_shapes
:         ђБ
BiasAdd/ReadVariableOpReadVariableOp!biasadd_readvariableop_resource_0",/job:localhost/replica:0/task:0/device:CPU:0*
dtype0*
_output_shapes	
:ђn
BiasAddBiasAddadd:z:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:         ђG
ConstConst*
value	B :*
dtype0*
_output_shapes
: Q
split/split_dimConst*
value	B :*
dtype0*
_output_shapes
: Х
splitSplitsplit/split_dim:output:0BiasAdd:output:0*
T0*
	num_split*`
_output_shapesN
L:         @:         @:         @:         @T
SigmoidSigmoidsplit:output:0*'
_output_shapes
:         @*
T0V
	Sigmoid_1Sigmoidsplit:output:1*
T0*'
_output_shapes
:         @Z
mulMulSigmoid_1:y:0placeholder_3*
T0*'
_output_shapes
:         @N
TanhTanhsplit:output:2*
T0*'
_output_shapes
:         @U
mul_1MulSigmoid:y:0Tanh:y:0*'
_output_shapes
:         @*
T0T
add_1AddV2mul:z:0	mul_1:z:0*
T0*'
_output_shapes
:         @V
	Sigmoid_2Sigmoidsplit:output:3*
T0*'
_output_shapes
:         @K
Tanh_1Tanh	add_1:z:0*
T0*'
_output_shapes
:         @Y
mul_2MulSigmoid_2:y:0
Tanh_1:y:0*'
_output_shapes
:         @*
T0Ї
$TensorArrayV2Write/TensorListSetItemTensorListSetItemplaceholder_1placeholder	mul_2:z:0*
element_dtype0*
_output_shapes
: I
add_2/yConst*
value	B :*
dtype0*
_output_shapes
: N
add_2AddV2placeholderadd_2/y:output:0*
T0*
_output_shapes
: I
add_3/yConst*
value	B :*
dtype0*
_output_shapes
: U
add_3AddV2while_loop_counteradd_3/y:output:0*
T0*
_output_shapes
: І
IdentityIdentity	add_3:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ю

Identity_1Identitywhile_maximum_iterations^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: Ї

Identity_2Identity	add_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: И

Identity_3Identity4TensorArrayV2Write/TensorListSetItem:output_handle:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*
_output_shapes
: ъ

Identity_4Identity	mul_2:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*'
_output_shapes
:         @*
T0ъ

Identity_5Identity	add_1:z:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^MatMul_1/ReadVariableOp*
T0*'
_output_shapes
:         @"
identityIdentity:output:0"!

identity_5Identity_5:output:0"F
 matmul_1_readvariableop_resource"matmul_1_readvariableop_resource_0"B
matmul_readvariableop_resource matmul_readvariableop_resource_0"ю
Ktensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensorMtensorarrayv2read_tensorlistgetitem_tensorarrayunstack_tensorlistfromtensor_0" 
strided_slicestrided_slice_0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"D
biasadd_readvariableop_resource!biasadd_readvariableop_resource_0"!

identity_3Identity_3:output:0"!

identity_4Identity_4:output:0*Q
_input_shapes@
>: : : : :         @:         @: : :::2.
MatMul/ReadVariableOpMatMul/ReadVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp22
MatMul_1/ReadVariableOpMatMul_1/ReadVariableOp: : : : : :	 :
 :  : : : "wL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*▓
serving_defaultъ
E

lstm_input7
serving_default_lstm_input:0         Z9
dense0
StatefulPartitionedCall:0         tensorflow/serving/predict*>
__saved_model_init_op%#
__saved_model_init_op

NoOp:╔н
¤*
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer

signatures
	variables
regularization_losses
		keras_api

trainable_variables
X__call__
*Y&call_and_return_all_conditional_losses
Z_default_save_signature"ѓ(
_tf_keras_sequentialс'{"model_config": {"class_name": "Sequential", "config": {"layers": [{"class_name": "LSTM", "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "activation": "tanh", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "batch_input_shape": [null, 90, 4], "activity_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "dropout": 0.0, "stateful": false, "unit_forget_bias": true, "recurrent_regularizer": null, "name": "lstm", "bias_constraint": null, "go_backwards": false, "implementation": 2, "kernel_regularizer": null, "return_sequences": true, "recurrent_constraint": null}}, {"class_name": "LSTM", "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "activation": "relu", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "unit_forget_bias": true, "activity_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "dropout": 0.0, "stateful": false, "recurrent_regularizer": null, "name": "lstm_1", "bias_constraint": null, "go_backwards": false, "implementation": 2, "kernel_regularizer": null, "return_sequences": false, "recurrent_constraint": null}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 30, "use_bias": true, "activity_regularizer": null}}], "name": "sequential"}}, "training_config": {"loss": "mae", "loss_weights": null, "metrics": [], "optimizer_config": {"class_name": "RMSprop", "config": {"name": "RMSprop", "decay": 0.0, "centered": false, "epsilon": 1e-07, "learning_rate": 0.0010000000474974513, "clipvalue": 1.0, "rho": 0.8999999761581421, "momentum": 0.0}}, "weighted_metrics": null, "sample_weight_mode": null}, "name": "sequential", "class_name": "Sequential", "dtype": "float32", "trainable": true, "input_spec": [{"class_name": "InputSpec", "config": {"ndim": 3, "max_ndim": null, "dtype": null, "axes": {}, "shape": [null, null, 4], "min_ndim": null}}], "expects_training_arg": true, "batch_input_shape": null, "config": {"layers": [{"class_name": "LSTM", "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "activation": "tanh", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "batch_input_shape": [null, 90, 4], "activity_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "dropout": 0.0, "stateful": false, "unit_forget_bias": true, "recurrent_regularizer": null, "name": "lstm", "bias_constraint": null, "go_backwards": false, "implementation": 2, "kernel_regularizer": null, "return_sequences": true, "recurrent_constraint": null}}, {"class_name": "LSTM", "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "activation": "relu", "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "unit_forget_bias": true, "activity_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "dropout": 0.0, "stateful": false, "recurrent_regularizer": null, "name": "lstm_1", "bias_constraint": null, "go_backwards": false, "implementation": 2, "kernel_regularizer": null, "return_sequences": false, "recurrent_constraint": null}}, {"class_name": "Dense", "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 30, "use_bias": true, "activity_regularizer": null}}], "name": "sequential"}, "keras_version": "2.2.4-tf", "backend": "tensorflow"}
»
	variables
regularization_losses
	keras_api
trainable_variables
[__call__
*\&call_and_return_all_conditional_losses"а
_tf_keras_layerє{"name": "lstm_input", "class_name": "InputLayer", "dtype": "float32", "batch_input_shape": [null, 90, 4], "trainable": true, "config": {"dtype": "float32", "batch_input_shape": [null, 90, 4], "name": "lstm_input", "sparse": false}, "expects_training_arg": true}
и

cell

state_spec
	variables
regularization_losses
	keras_api
trainable_variables
]__call__
*^&call_and_return_all_conditional_losses"ј	
_tf_keras_layerЗ{"name": "lstm", "class_name": "LSTM", "dtype": "float32", "batch_input_shape": [null, 90, 4], "trainable": true, "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "batch_input_shape": [null, 90, 4], "dropout": 0.0, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "activation": "tanh", "stateful": false, "activity_regularizer": null, "unit_forget_bias": true, "recurrent_regularizer": null, "name": "lstm", "bias_constraint": null, "go_backwards": false, "implementation": 2, "kernel_regularizer": null, "return_sequences": true, "recurrent_constraint": null}, "input_spec": [{"class_name": "InputSpec", "config": {"ndim": 3, "max_ndim": null, "dtype": null, "axes": {}, "shape": [null, null, 4], "min_ndim": null}}], "expects_training_arg": true}
љ

cell

state_spec
	variables
regularization_losses
	keras_api
trainable_variables
___call__
*`&call_and_return_all_conditional_losses"у
_tf_keras_layer═{"name": "lstm_1", "class_name": "LSTM", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"recurrent_activation": "sigmoid", "dtype": "float32", "trainable": true, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "use_bias": true, "bias_regularizer": null, "return_state": false, "unroll": false, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "unit_forget_bias": true, "dropout": 0.0, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "kernel_constraint": null, "time_major": false, "activation": "relu", "stateful": false, "activity_regularizer": null, "recurrent_regularizer": null, "name": "lstm_1", "bias_constraint": null, "go_backwards": false, "implementation": 2, "kernel_regularizer": null, "return_sequences": false, "recurrent_constraint": null}, "input_spec": [{"class_name": "InputSpec", "config": {"ndim": 3, "max_ndim": null, "dtype": null, "axes": {}, "shape": [null, null, 64], "min_ndim": null}}], "expects_training_arg": true}
№

kernel
bias
	variables
regularization_losses
	keras_api
 trainable_variables
a__call__
*b&call_and_return_all_conditional_losses"╩
_tf_keras_layer░{"name": "dense", "class_name": "Dense", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "dense", "kernel_constraint": null, "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "linear", "trainable": true, "kernel_regularizer": null, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 30, "use_bias": true, "activity_regularizer": null}, "input_spec": {"class_name": "InputSpec", "config": {"ndim": null, "max_ndim": null, "dtype": null, "axes": {"-1": 64}, "shape": null, "min_ndim": 2}}, "expects_training_arg": false}
ф
!iter
	"decay
#learning_rate
$momentum
%rho	rmsP	rmsQ	&rmsR	'rmsS	(rmsT	)rmsU	*rmsV	+rmsW"
	optimizer
,
cserving_default"
signature_map
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
и

,layers

trainable_variables
	variables
-layer_regularization_losses
.metrics
regularization_losses
/non_trainable_variables
X__call__
Z_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
X
&0
'1
(2
)3
*4
+5
6
7"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
џ

0layers
trainable_variables
	variables
1layer_regularization_losses
2metrics
regularization_losses
3non_trainable_variables
[__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
Ь

&kernel
'recurrent_kernel
(bias
4	variables
5regularization_losses
6	keras_api
7trainable_variables
d__call__
*e&call_and_return_all_conditional_losses"│
_tf_keras_layerЎ{"name": "lstm_cell", "class_name": "LSTMCell", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"recurrent_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "lstm_cell", "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "tanh", "trainable": true, "kernel_constraint": null, "implementation": 2, "kernel_regularizer": null, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "unit_forget_bias": true, "use_bias": true, "recurrent_constraint": null, "dropout": 0.0, "recurrent_activation": "sigmoid"}, "expects_training_arg": true}
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
џ

8layers
trainable_variables
	variables
9layer_regularization_losses
:metrics
regularization_losses
;non_trainable_variables
]__call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
5
&0
'1
(2"
trackable_list_wrapper
Ы

)kernel
*recurrent_kernel
+bias
<	variables
=regularization_losses
>	keras_api
?trainable_variables
f__call__
*g&call_and_return_all_conditional_losses"и
_tf_keras_layerЮ{"name": "lstm_cell_1", "class_name": "LSTMCell", "dtype": "float32", "batch_input_shape": null, "trainable": true, "config": {"recurrent_regularizer": null, "recurrent_dropout": 0.0, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "name": "lstm_cell_1", "bias_regularizer": null, "bias_constraint": null, "dtype": "float32", "activation": "relu", "trainable": true, "kernel_constraint": null, "implementation": 2, "kernel_regularizer": null, "recurrent_initializer": {"class_name": "Orthogonal", "config": {"seed": null, "gain": 1.0}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "units": 64, "unit_forget_bias": true, "use_bias": true, "recurrent_constraint": null, "dropout": 0.0, "recurrent_activation": "sigmoid"}, "expects_training_arg": true}
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
џ

@layers
trainable_variables
	variables
Alayer_regularization_losses
Bmetrics
regularization_losses
Cnon_trainable_variables
___call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
5
)0
*1
+2"
trackable_list_wrapper
:@2dense/kernel
:2
dense/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
џ

Dlayers
 trainable_variables
	variables
Elayer_regularization_losses
Fmetrics
regularization_losses
Gnon_trainable_variables
a__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
:	 (2RMSprop/iter
: (2RMSprop/decay
: (2RMSprop/learning_rate
: (2RMSprop/momentum
: (2RMSprop/rho
:	ђ2lstm/kernel
(:&	@ђ2lstm/recurrent_kernel
:ђ2	lstm/bias
 :	@ђ2lstm_1/kernel
*:(	@ђ2lstm_1/recurrent_kernel
:ђ2lstm_1/bias
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
&0
'1
(2"
trackable_list_wrapper
 "
trackable_list_wrapper
џ

Hlayers
7trainable_variables
4	variables
Ilayer_regularization_losses
Jmetrics
5regularization_losses
Knon_trainable_variables
d__call__
*e&call_and_return_all_conditional_losses
&e"call_and_return_conditional_losses"
_generic_user_object
5
&0
'1
(2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
)0
*1
+2"
trackable_list_wrapper
 "
trackable_list_wrapper
џ

Llayers
?trainable_variables
<	variables
Mlayer_regularization_losses
Nmetrics
=regularization_losses
Onon_trainable_variables
f__call__
*g&call_and_return_all_conditional_losses
&g"call_and_return_conditional_losses"
_generic_user_object
5
)0
*1
+2"
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(:&@2RMSprop/dense/kernel/rms
": 2RMSprop/dense/bias/rms
(:&	ђ2RMSprop/lstm/kernel/rms
2:0	@ђ2!RMSprop/lstm/recurrent_kernel/rms
": ђ2RMSprop/lstm/bias/rms
*:(	@ђ2RMSprop/lstm_1/kernel/rms
4:2	@ђ2#RMSprop/lstm_1/recurrent_kernel/rms
$:"ђ2RMSprop/lstm_1/bias/rms
Ш2з
*__inference_sequential_layer_call_fn_12616
*__inference_sequential_layer_call_fn_11263
*__inference_sequential_layer_call_fn_11294
*__inference_sequential_layer_call_fn_12629└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
Р2▀
E__inference_sequential_layer_call_and_return_conditional_losses_11233
E__inference_sequential_layer_call_and_return_conditional_losses_11959
E__inference_sequential_layer_call_and_return_conditional_losses_12603
E__inference_sequential_layer_call_and_return_conditional_losses_11216└
и▓│
FullArgSpec1
args)џ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsџ
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
С2р
__inference__wrapped_model_7306й
І▓Є
FullArgSpec
argsџ 
varargsjargs
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *-б*
(і%

lstm_input         Z
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
╠2╔к
й▓╣
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkwjkwargs
defaultsџ 

kwonlyargsџ

jtraining%
kwonlydefaultsф

trainingp 
annotationsф *
 
з2­
$__inference_lstm_layer_call_fn_12645
$__inference_lstm_layer_call_fn_12637
$__inference_lstm_layer_call_fn_13603
$__inference_lstm_layer_call_fn_13595Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
▀2▄
?__inference_lstm_layer_call_and_return_conditional_losses_14076
?__inference_lstm_layer_call_and_return_conditional_losses_13587
?__inference_lstm_layer_call_and_return_conditional_losses_13116
?__inference_lstm_layer_call_and_return_conditional_losses_14549Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ч2Э
&__inference_lstm_1_layer_call_fn_15245
&__inference_lstm_1_layer_call_fn_14903
&__inference_lstm_1_layer_call_fn_14895
&__inference_lstm_1_layer_call_fn_15253Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
у2С
A__inference_lstm_1_layer_call_and_return_conditional_losses_15070
A__inference_lstm_1_layer_call_and_return_conditional_losses_14718
A__inference_lstm_1_layer_call_and_return_conditional_losses_14887
A__inference_lstm_1_layer_call_and_return_conditional_losses_15237Н
╠▓╚
FullArgSpecB
args:џ7
jself
jinputs
jmask

jtraining
jinitial_state
varargs
 
varkw
 
defaultsџ

 
p 

 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
¤2╠
%__inference_dense_layer_call_fn_15260б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
Ж2у
@__inference_dense_layer_call_and_return_conditional_losses_15270б
Ў▓Ћ
FullArgSpec
argsџ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsџ 
kwonlydefaults
 
annotationsф *
 
5B3
#__inference_signature_wrapper_11313
lstm_input
─2┴Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
─2┴Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
ъ2Џ
+__inference_lstm_cell_1_layer_call_fn_15350
+__inference_lstm_cell_1_layer_call_fn_15364Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 
н2Л
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_15336
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_15303Й
х▓▒
FullArgSpec3
args+џ(
jself
jinputs
jstates

jtraining
varargs
 
varkw
 
defaultsџ
p 

kwonlyargsџ 
kwonlydefaultsф 
annotationsф *
 і
&__inference_lstm_1_layer_call_fn_14895`)*+?б<
5б2
$і!
inputs         Z@

 
p

 
ф "і         @е
#__inference_signature_wrapper_11313ђ&'()*+EбB
б 
;ф8
6

lstm_input(і%

lstm_input         Z"-ф*
(
denseі
dense         ┬
A__inference_lstm_1_layer_call_and_return_conditional_losses_15070})*+OбL
EбB
4џ1
/і,
inputs/0                  @

 
p

 
ф "%б"
і
0         @
џ ╚
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_15303§)*+ђб}
vбs
 і
inputs         @
KбH
"і
states/0         @
"і
states/1         @
p
ф "sбp
iбf
і
0/0         @
EџB
і
0/1/0         @
і
0/1/1         @
џ Њ
*__inference_sequential_layer_call_fn_11263e&'()*+?б<
5б2
(і%

lstm_input         Z
p

 
ф "і         џ
&__inference_lstm_1_layer_call_fn_15245p)*+OбL
EбB
4џ1
/і,
inputs/0                  @

 
p

 
ф "і         @џ
&__inference_lstm_1_layer_call_fn_15253p)*+OбL
EбB
4џ1
/і,
inputs/0                  @

 
p 

 
ф "і         @Ј
*__inference_sequential_layer_call_fn_12616a&'()*+;б8
1б.
$і!
inputs         Z
p

 
ф "і         ї
$__inference_lstm_layer_call_fn_13603d&'(?б<
5б2
$і!
inputs         Z

 
p 

 
ф "і         Z@╗
E__inference_sequential_layer_call_and_return_conditional_losses_11216r&'()*+?б<
5б2
(і%

lstm_input         Z
p

 
ф "%б"
і
0         
џ Њ
*__inference_sequential_layer_call_fn_11294e&'()*+?б<
5б2
(і%

lstm_input         Z
p 

 
ф "і         ╚
F__inference_lstm_cell_1_layer_call_and_return_conditional_losses_15336§)*+ђб}
vбs
 і
inputs         @
KбH
"і
states/0         @
"і
states/1         @
p 
ф "sбp
iбf
і
0/0         @
EџB
і
0/1/0         @
і
0/1/1         @
џ Ј
*__inference_sequential_layer_call_fn_12629a&'()*+;б8
1б.
$і!
inputs         Z
p 

 
ф "і         ▓
A__inference_lstm_1_layer_call_and_return_conditional_losses_14887m)*+?б<
5б2
$і!
inputs         Z@

 
p 

 
ф "%б"
і
0         @
џ и
E__inference_sequential_layer_call_and_return_conditional_losses_11959n&'()*+;б8
1б.
$і!
inputs         Z
p

 
ф "%б"
і
0         
џ ╗
E__inference_sequential_layer_call_and_return_conditional_losses_11233r&'()*+?б<
5б2
(і%

lstm_input         Z
p 

 
ф "%б"
і
0         
џ а
@__inference_dense_layer_call_and_return_conditional_losses_15270\/б,
%б"
 і
inputs         @
ф "%б"
і
0         
џ ▓
A__inference_lstm_1_layer_call_and_return_conditional_losses_14718m)*+?б<
5б2
$і!
inputs         Z@

 
p

 
ф "%б"
і
0         @
џ ┤
?__inference_lstm_layer_call_and_return_conditional_losses_14549q&'(?б<
5б2
$і!
inputs         Z

 
p 

 
ф ")б&
і
0         Z@
џ x
%__inference_dense_layer_call_fn_15260O/б,
%б"
 і
inputs         @
ф "і         Ю
+__inference_lstm_cell_1_layer_call_fn_15350ь)*+ђб}
vбs
 і
inputs         @
KбH
"і
states/0         @
"і
states/1         @
p
ф "cб`
і
0         @
Aџ>
і
1/0         @
і
1/1         @і
&__inference_lstm_1_layer_call_fn_14903`)*+?б<
5б2
$і!
inputs         Z@

 
p 

 
ф "і         @ї
$__inference_lstm_layer_call_fn_13595d&'(?б<
5б2
$і!
inputs         Z

 
p

 
ф "і         Z@Ц
$__inference_lstm_layer_call_fn_12637}&'(OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "%і"                  @┬
A__inference_lstm_1_layer_call_and_return_conditional_losses_15237})*+OбL
EбB
4џ1
/і,
inputs/0                  @

 
p 

 
ф "%б"
і
0         @
џ Ц
$__inference_lstm_layer_call_fn_12645}&'(OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "%і"                  @и
E__inference_sequential_layer_call_and_return_conditional_losses_12603n&'()*+;б8
1б.
$і!
inputs         Z
p 

 
ф "%б"
і
0         
џ ╬
?__inference_lstm_layer_call_and_return_conditional_losses_13116і&'(OбL
EбB
4џ1
/і,
inputs/0                  

 
p

 
ф "2б/
(і%
0                  @
џ Ю
+__inference_lstm_cell_1_layer_call_fn_15364ь)*+ђб}
vбs
 і
inputs         @
KбH
"і
states/0         @
"і
states/1         @
p 
ф "cб`
і
0         @
Aџ>
і
1/0         @
і
1/1         @┤
?__inference_lstm_layer_call_and_return_conditional_losses_14076q&'(?б<
5б2
$і!
inputs         Z

 
p

 
ф ")б&
і
0         Z@
џ Ћ
__inference__wrapped_model_7306r&'()*+7б4
-б*
(і%

lstm_input         Z
ф "-ф*
(
denseі
dense         ╬
?__inference_lstm_layer_call_and_return_conditional_losses_13587і&'(OбL
EбB
4џ1
/і,
inputs/0                  

 
p 

 
ф "2б/
(і%
0                  @
џ 