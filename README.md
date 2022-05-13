# A point Cloud-based network verifier (Initial Repo) 

## Setup Virtual Environment

```ruby
$
$ !python3 -m venv .venv
$ . .venv/bin/activate
```
Open PCV_Onnx_shared_v1.ipynb file in google colab or any local python environment and execute all cells.
Be sure to enable GPU.
If you face any issue to load the PTH and ONNX file, generate the file once again and check the path.

## Sample guidelines with DNNV. For future references, not needed for PCV
## Install DNNV

```ruby
$ git clone https://github.com/dlshriver/DNNV.git
$ cd DNNV
$ !pip3 install .
```
## Install Dependencies
$ !dnnv_manage install planet mipverify neurify eran bab marabou nnenum verinet
reluplex has build error problem
## Verify mnist onnx model with ERAN properties

```ruby
$ git clone https://github.com/arupcsedu/NeuralNetworkVerification.git
$ cd NeuralNetworkVerification/ERAN-MNIST
$ dnnv property_0.py --network N pytorch_mnist.onnx
```

## Output

```ruby
Verifying property:
Forall(x_, ((([[[-0.008 -0.008 ... -0.008 -0.008] [-0.008 -0.008 ... -0.008 -0.008] ... [-0.008 -0.008 ... -0.008 -0.008] [-0.008 -0.008 ... -0.008 -0.008]]] < (0.1307 + (0.3081 * x_))) & ((0.1307 + (0.3081 * x_)) < [[[0.008 0.008 ... 0.008 0.008] [0.008 0.008 ... 0.008 0.008] ... [0.008 0.008 ... 0.008 0.008] [0.008 0.008 ... 0.008 0.008]]]) & (0 < (0.1307 + (0.3081 * x_))) & ((0.1307 + (0.3081 * x_)) < 1)) ==> (numpy.argmax(N(x_)) == 7)))

Verifying Networks:
N:
Input_0                         : Input([  1 784], dtype=float32)
Gemm_0                          : Gemm(Input_0, ndarray(shape=(128, 784)), ndarray(shape=(128,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
Relu_0                          : Relu(Gemm_0)
Gemm_1                          : Gemm(Relu_0, ndarray(shape=(64, 128)), ndarray(shape=(64,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
Relu_1                          : Relu(Gemm_1)
Gemm_2                          : Gemm(Relu_1, ndarray(shape=(10, 64)), ndarray(shape=(10,)), transpose_a=0, transpose_b=1, alpha=1.000000, beta=1.000000)
LogSoftmax_0                    : LogSoftmax(Gemm_2, axis=1)
```
