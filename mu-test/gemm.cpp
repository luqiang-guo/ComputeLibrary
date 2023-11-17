#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"
#include "arm_compute/runtime/Scheduler.h"
#include <cstddef>
#include <iostream>

using namespace arm_compute;


template <typename T>
void init_sgemm_output(T &dst, T &src0, T &src1, arm_compute::DataType dt)
{
    dst.allocator()->init(TensorInfo(TensorShape(src1.info()->dimension(0), src0.info()->dimension(1), src0.info()->dimension(2)), 1, dt));
}


int main()
{

    arm_compute::Scheduler::get().set_num_threads(1);
    // Create tensors
    size_t M = 23;
    size_t N = 35;
    size_t K = 7;

    arm_compute::Tensor A;
    arm_compute::Tensor B;
    
    // Initialize tensors
    A.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(K, M), 1, arm_compute::DataType::F32));
    B.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(N, K), 1, arm_compute::DataType::F32));


    // Create output tensor
    arm_compute::Tensor C;
    // C.allocator()->init(arm_compute::TensorInfo(arm_compute::TensorShape(4, 4), 1, arm_compute::DataType::F32));

    init_sgemm_output(C, A, B, arm_compute::DataType::F32);
    // Create and configure the function
    arm_compute::NEGEMM sgemm;
    float   alpha = 1.0, beta = 0.0;
    sgemm.configure(&A, &B, nullptr, &C, alpha, beta);

    // Allocate tensors and run the function

    A.allocator()->allocate();
    B.allocator()->allocate();
    C.allocator()->allocate();

    float *a_ptr = (float *)A.buffer();
    float *b_ptr = (float *)B.buffer();

    for(size_t i = 0; i < K*M; i++) {
        a_ptr[i] = i*0.1;
    }

    for(size_t i = 0; i < K*N; i++) {
        b_ptr[i] = i*0.1;
    }

    sgemm.run();

    float *c_ptr = (float *)C.buffer();
    
    // for (int i = 0; i < M; ++i)
    // {
    //     for(int j = 0; j < N; j++)
    //     {
    //         printf("%.1f, ", c_ptr[i*N + j]);
    //     }
    //     printf("\n");
    // }
    
    return 0;
}
