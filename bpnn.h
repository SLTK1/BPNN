#ifndef _BPNN_H_
#define _BPNN_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "readfile.h"
typedef struct NN
{
    int hidden_l;  //神经网络隐藏层数
    int layer;  //总层数

    //int input_n;    //输入层节点数
    //int output_n;   //输出层节点数
    //int *hidden_n;  //隐藏层节点数
    int *node_num;  //各层节点数

    //double *input_node; //输入层节点
    //double *output_node;    //输出层节点
    //double **hidden_node;   //隐藏层节点

    double **node;

    double ***weight;   //权值
    double **d;    //反向误差
}NN;

double sigmoid(double x);
void Set_nn(NN* nn, int N);
void Create_nn(NN* nn);
void Show_nn(NN* nn);
void Show_nn_d(NN* nn);
void Show_result(NN* nn);
void Init_nn(NN* nn, int flag);
void foward_propagation(NN* nn);
void back_propagation(NN* nn, int step, int sample_n, int output_sample_dim, double* output, double learning_rate);
void update_input_layer(NN* nn, int step, int sample_n, int input_sample_dim, double* input);
double cal_loss(NN* nn, int step, int sample_n, int output_sample_dim, double* output);
double train(NN* nn, int step_num, double learning_rate, double* input, double* output, int sample_n, int input_sample_dim, int output_sample_dim, double er);
void test(NN* nn, double* input, double* output, int test_n, int input_sample_dim, int output_sample_dim);
#endif // _BPNN_H_
