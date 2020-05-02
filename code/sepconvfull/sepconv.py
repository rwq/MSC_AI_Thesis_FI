import cupy
import torch
import re

kernel_Sepconv_updateOutput = '''
    extern "C" __global__ void kernel_Sepconv_updateOutput(
        const int n,
        const float* input,
        const float* vertical,
        const float* horizontal,
        float* output
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float dblOutput = 0.0;

        const int intSample = ( intIndex / SIZE_3(output) / SIZE_2(output) / SIZE_1(output) ) % SIZE_0(output);
        const int intDepth  = ( intIndex / SIZE_3(output) / SIZE_2(output)                  ) % SIZE_1(output);
        const int intY      = ( intIndex / SIZE_3(output)                                   ) % SIZE_2(output);
        const int intX      = ( intIndex                                                    ) % SIZE_3(output);

        for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY += 1) {
        for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX += 1) {
        dblOutput += VALUE_4(input, intSample, intDepth, intY + intFilterY, intX + intFilterX) * VALUE_4(vertical, intSample, intFilterY, intY, intX) * VALUE_4(horizontal, intSample, intFilterX, intY, intX);
            }
        }

        output[intIndex] = dblOutput;
    } }
'''

kernel_SeparableConvolution_updateGradVertical = '''
    extern "C" __global__ void kernel_SeparableConvolution_updateGradVertical(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* horizontal,
        float* gradVertical
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;
        float c = 0.0;
    
        const int intBatch   = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical) / SIZE_1(gradVertical) ) % SIZE_0(gradVertical);
        const int intFilterY = ( intIndex / SIZE_3(gradVertical) / SIZE_2(gradVertical)                        ) % SIZE_1(gradVertical);
        const int intY       = ( intIndex / SIZE_3(gradVertical)                                               ) % SIZE_2(gradVertical);
        const int intX       = ( intIndex                                                                      ) % SIZE_3(gradVertical);
    
        for (int intFilterX = 0; intFilterX < SIZE_1(horizontal); intFilterX++) 
        {
            float product = VALUE_4(gradLoss, intBatch, 0, intY, intX)*              // channel 0
            VALUE_4(input, intBatch, 0, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(horizontal, intBatch, intFilterX, intY, intX) +
            VALUE_4(gradLoss, intBatch, 1, intY, intX)*                          // channel 1     
            VALUE_4(input, intBatch, 1, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(horizontal, intBatch, intFilterX, intY, intX) +
            VALUE_4(gradLoss, intBatch, 2, intY, intX)*                          // channel 2
            VALUE_4(input, intBatch, 2, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(horizontal, intBatch, intFilterX, intY, intX);

            floatOutput += product;
        }
    
        gradVertical[intIndex] = floatOutput;
    } }
'''

kernel_SeparableConvolution_updateGradHorizontal = '''
    extern "C" __global__ void kernel_SeparableConvolution_updateGradHorizontal(
        const int n,
        const float* gradLoss,
        const float* input,
        const float* vertical,
        float* gradHorizontal
    ) { for (int intIndex = (blockIdx.x * blockDim.x) + threadIdx.x; intIndex < n; intIndex += blockDim.x * gridDim.x) {
        float floatOutput = 0.0;
        float c = 0.0;
    
        const int intBatch   = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal) / SIZE_1(gradHorizontal) ) % SIZE_0(gradHorizontal);
        const int intFilterX = ( intIndex / SIZE_3(gradHorizontal) / SIZE_2(gradHorizontal)                          ) % SIZE_1(gradHorizontal);
        const int intY       = ( intIndex / SIZE_3(gradHorizontal)                                                   ) % SIZE_2(gradHorizontal);
        const int intX       = ( intIndex                                                                            ) % SIZE_3(gradHorizontal);
    
        for (int intFilterY = 0; intFilterY < SIZE_1(vertical); intFilterY++)
        {
            float product = VALUE_4(gradLoss, intBatch, 0, intY, intX)*             // channel 0
            VALUE_4(input, intBatch, 0, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(vertical, intBatch, intFilterY, intY, intX) + 
            VALUE_4(gradLoss, intBatch, 1, intY, intX)*                         // channel 1
            VALUE_4(input, intBatch, 1, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(vertical, intBatch, intFilterY, intY, intX) + 
            VALUE_4(gradLoss, intBatch, 2, intY, intX)*                         // channel 2
            VALUE_4(input, intBatch, 2, intY + intFilterY, intX + intFilterX)* 
            VALUE_4(vertical, intBatch, intFilterY, intY, intX);
    
            float y = product - c;
            float t = floatOutput + y;
            c = (t - floatOutput) - y;
            floatOutput = t;
        }
    
        gradHorizontal[intIndex] = floatOutput;
    } }
'''


def cupy_kernel(strFunction, objectVariables):
    strKernel = globals()[strFunction]

    while True:
        objectMatch = re.search('(SIZE_)([0-4])(\()([^\)]*)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArg = int(objectMatch.group(2))

        strTensor = objectMatch.group(4)
        intSizes = objectVariables[strTensor].size()

        strKernel = strKernel.replace(objectMatch.group(), str(intSizes[intArg]))
    # end

    while True:
        objectMatch = re.search('(VALUE_)([0-4])(\()([^\)]+)(\))', strKernel)

        if objectMatch is None:
            break
        # end

        intArgs = int(objectMatch.group(2))
        strArgs = objectMatch.group(4).split(',')

        strTensor = strArgs[0]
        intStrides = objectVariables[strTensor].stride()
        strIndex = ['((' + strArgs[intArg + 1].replace('{', '(').replace('}', ')').strip() + ')*' + str(intStrides[intArg]) + ')' for intArg in range(intArgs)]

        strKernel = strKernel.replace(objectMatch.group(0), strTensor + '[' + str.join('+', strIndex) + ']')
    # end

    return strKernel


# end

@cupy.util.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)


# end


class FunctionSepconv(torch.nn.Module):

    def __init__(self):
        super(FunctionSepconv, self).__init__()

    def forward(self, input, vertical, horizontal):
        return _FunctionSepconv.apply(input, vertical, horizontal)


class _FunctionSepconv(torch.autograd.Function):
    def __init__(self):
        super(_FunctionSepconv, self).__init__()

    # end
    @staticmethod
    def forward(self, input, vertical, horizontal):
        self.save_for_backward(input, vertical, horizontal)

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert (intInputHeight - intFilterSize == intOutputHeight - 1)
        assert (intInputWidth - intFilterSize == intOutputWidth - 1)

        assert (input.is_contiguous() == True)
        assert (vertical.is_contiguous() == True)
        assert (horizontal.is_contiguous() == True)

        output = input.new_zeros(intSample, intInputDepth, intOutputHeight, intOutputWidth)

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            n = output.nelement()
            cupy_launch('kernel_Sepconv_updateOutput', cupy_kernel('kernel_Sepconv_updateOutput', {
                'input': input,
                'vertical': vertical,
                'horizontal': horizontal,
                'output': output
            }))(
                grid=tuple([int((n + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n, input.data_ptr(), vertical.data_ptr(), horizontal.data_ptr(), output.data_ptr()],
                stream=Stream
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return output

    # end
    @staticmethod
    def backward(self, gradOutput):
        input, vertical, horizontal = self.saved_tensors

        intSample = input.size(0)
        intInputDepth = input.size(1)
        intInputHeight = input.size(2)
        intInputWidth = input.size(3)
        intFilterSize = min(vertical.size(1), horizontal.size(1))
        intOutputHeight = min(vertical.size(2), horizontal.size(2))
        intOutputWidth = min(vertical.size(3), horizontal.size(3))

        assert (intInputHeight - intFilterSize == intOutputHeight - 1)
        assert (intInputWidth - intFilterSize == intOutputWidth - 1)

        assert (gradOutput.is_contiguous() == True)

        gradInput = input.new_zeros(intSample, intInputDepth, intInputHeight, intInputWidth) if self.needs_input_grad[0] == True else None
        gradVertical = input.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth) if self.needs_input_grad[1] == True else None
        gradHorizontal = input.new_zeros(intSample, intFilterSize, intOutputHeight, intOutputWidth) if self.needs_input_grad[2] == True else None

        if input.is_cuda == True:
            class Stream:
                ptr = torch.cuda.current_stream().cuda_stream

            # end

            # vertical grad
            n_v = gradVertical.nelement()
            cupy_launch('kernel_SeparableConvolution_updateGradVertical', cupy_kernel('kernel_SeparableConvolution_updateGradVertical', {
                'gradLoss': gradOutput,
                'input': input,
                'horizontal': horizontal,
                'gradVertical': gradVertical
            }))(
                grid=tuple([int((n_v + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n_v, gradOutput.data_ptr(), input.data_ptr(), horizontal.data_ptr(), gradVertical.data_ptr()],
                stream=Stream
            )

            # horizontal grad
            n_h = gradHorizontal.nelement()
            cupy_launch('kernel_SeparableConvolution_updateGradHorizontal', cupy_kernel('kernel_SeparableConvolution_updateGradHorizontal', {
                'gradLoss': gradOutput,
                'input': input,
                'vertical': vertical,
                'gradHorizontal': gradHorizontal
            }))(
                grid=tuple([int((n_h + 512 - 1) / 512), 1, 1]),
                block=tuple([512, 1, 1]),
                args=[n_h, gradOutput.data_ptr(), input.data_ptr(), vertical.data_ptr(), gradHorizontal.data_ptr()],
                stream=Stream
            )

        elif input.is_cuda == False:
            raise NotImplementedError()

        # end

        return gradInput, gradVertical, gradHorizontal


# end


# end


