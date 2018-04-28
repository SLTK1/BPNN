#include <stdio.h>
#include <stdlib.h>
#include "bpnn.h"
#include "readfile.h"
#define M 128
#define N1 7
#define M_train 128
int main()
{
    NN bpnn;
    int sample_n, input_sample_dim, output_sample_dim, test_n;
    //double input[8] = {0, 1, 1, 0, 1, 1, 0, 0};
    //double output[4] = {1, 1, 0, 0};
    //double input_test[8] = {0, 1, 1, 0, 1, 1, 0, 0};
    //double output_test[4] = {1, 1, 0, 0};

    double Loss, learning_rate = 8;
    double er = 0.001;
    int N = N1;
    int step_num = 1000000;

    int a[M*N1];
    int b[M];
    double input[M*N1];
    double output[M*N1];
    double input_test[M];
    double output_test[M];
    /*
    sample_n = 4;
    test_n = 4;
    input_sample_dim = 2;
    output_sample_dim = 1;
    */

    sample_n = M;
    test_n = 1;
    input_sample_dim = N1;
    output_sample_dim = 1;

    readfile(a, b, M, N1);
    divide_data(a, b, input, output, input_test, output_test, M, N1, M_train);

    Set_nn(&bpnn, N);
    Create_nn(&bpnn);

    Loss = train(&bpnn, step_num, learning_rate, input, output, sample_n, input_sample_dim, output_sample_dim, er);
    Show_nn(&bpnn);
    test(&bpnn, input_test, output_test, test_n, input_sample_dim, output_sample_dim);
    //Show_nn(&bpnn);

    return 0;
}
