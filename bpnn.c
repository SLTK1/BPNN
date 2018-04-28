#include "bpnn.h"

double sigmoid(double x)
{
    x = 1.0/(1 + exp(-x));
    return x;
}

void Set_nn(NN* nn, int N)
{
    int i;
    printf("input the number of hidden layer: ");
    scanf("%d", &nn->hidden_l);
    nn->layer = nn->hidden_l + 2;
    nn->node_num = (int*)malloc(sizeof(int) * nn->layer);
    printf("input the node number of input layer: ");
    scanf("%d", &nn->node_num[0]);
    if(nn->node_num[0] != N)
    {
        printf("Error input\n");
        exit(0);
    }
    for(i = 0; i < nn->hidden_l; ++i)
    {
        printf("input the node number of hidden layer %d: ", i+1);
        scanf("%d", &nn->node_num[i+1]);
    }
    printf("input the node number of output layer: ");
    scanf("%d", &nn->node_num[i+1]);
    printf("set function\n");
}

void Create_nn(NN* nn)
{
    int i, j;
    nn->node = (double**)malloc(sizeof(double*) * nn->layer);
    for(i = 0; i < nn->layer-1; ++i)
        nn->node[i] = (double*)malloc(sizeof(double) * (nn->node_num[i] + 1));
    nn->node[i] = (double*)malloc(sizeof(double) * nn->node_num[i]);
    nn->weight = (double***)malloc(sizeof(double**) * (nn->layer - 1));
    for(i = 0; i < nn->layer-1; ++i)
    {
        nn->weight[i] = (double**)malloc(sizeof(double*) * (nn->node_num[i]+1));
        for(j = 0; j < nn->node_num[i] + 1; ++j)
            nn->weight[i][j] = (double*)malloc(sizeof(double) * (nn->node_num[i+1]));
    }
    nn->d = (double**)malloc(sizeof(double*) * (nn->layer - 1));
    for(i = 0; i < nn->layer-1; ++i)
        nn->d[i] = (double*)malloc(sizeof(double) * nn->node_num[i+1]);
    printf("Create function\n");
}

void Show_nn(NN* nn)
{
    int i, j, k;
    printf("the number of hidden layer is %d\n", nn->hidden_l);
    printf("the number of input layer node is %d\n", nn->node_num[0]);
    for(i = 0; i < nn->hidden_l; ++i)
        printf("the number of hidden layer %d node is %d\n", i+1, nn->node_num[i+1]);
    printf("the number of output layer node is %d\n", nn->node_num[i+1]);
    for(i = 0; i < nn->layer-1; ++i)
        for(j = 0; j <= nn->node_num[i]; ++j)
            printf("node[%d][%d] = %f\n", i, j, nn->node[i][j]);
    for(j = 0; j < nn->node_num[i]; ++j)
        printf("node[%d][%d] = %f\n", i, j, nn->node[i][j]);
    for(i = 0; i < nn->layer-1; ++i)
        for(j = 0; j <= nn->node_num[i]; ++j)
            for(k = 0; k < nn->node_num[i+1]; ++k)
                printf("weight[%d][%d][%d] = %f\n", i, j, k, nn->weight[i][j][k]);
}

void Show_nn_d(NN* nn)
{
    int i, j;
    for(i = 0; i < nn->layer-1; ++i)
        for(j = 0; j < nn->node_num[i+1]; ++j)
            printf("d[%d][%d] = %f\n", i, j, nn->d[i][j]);
}

void Show_result(NN* nn)
{
    int i;
    for(i = 0; i < nn->node_num[0]; ++i)
        printf("%f ", nn->node[0][i]);
    for(i = 0; i < nn->node_num[nn->layer-1]; ++i)
        printf("%f\n", nn->node[nn->layer-1][i]);
}

void Init_nn(NN* nn, int flag)
{
    int i, j, k;
    for(i = 0; i < nn->layer; ++i)
        for(j = 0; j < nn->node_num[i]; ++j)
            nn->node[i][j] = 0;
    for(i = 0; i < nn->layer-1; ++i)
        nn->node[i][nn->node_num[i]] = 1;
    for(i = 0; i < nn->layer-1; ++i)
            for(j = 0; j < nn->node_num[i+1]; ++j)
                nn->d[i][j] = 0;
    if(flag == 0)
    {
        srand((unsigned)time(NULL));
        for(i = 0; i < nn->layer-1; ++i)
            for(j = 0; j <= nn->node_num[i]; ++j)
                for(k = 0; k < nn->node_num[i+1]; ++k)
                {
                    nn->weight[i][j][k] = (double)rand()/RAND_MAX;
                    nn->weight[i][j][k] -= 0.5;
                    nn->weight[i][j][k] *= 2;
                    ///nn->weight[i][j][k] *= 0.2;
                }
    }
    //printf("Init function\n");
}

void foward_propagation(NN* nn)
{
    int i, j, k;
    for(i = 1; i < nn->layer; ++i)
        for(j = 0; j < nn->node_num[i]; ++j)
        {
            for(k = 0; k <= nn->node_num[i-1]; ++k)
                nn->node[i][j] += nn->node[i-1][k] * nn->weight[i-1][k][j];
            nn->node[i][j] = sigmoid(nn->node[i][j]);
        }
}

void back_propagation(NN* nn, int step, int sample_n, int output_sample_dim, double* output, double learning_rate)
{
    int i, j, k;
    double temp;
    //calculate d
    for(i = 0; i < nn->node_num[nn->layer-1]; i++)
        nn->d[nn->layer-2][i] = (output[step%sample_n * output_sample_dim + i] - nn->node[nn->layer-1][i]) * nn->node[nn->layer-1][i] * (1 - nn->node[nn->layer-1][i]);
    for(i = nn->layer-2; i > 0; --i)
    {
        for(j = 0; j < nn->node_num[i]; ++j)
        {
            temp = 0;
            for(k = 0; k < nn->node_num[i+1]; ++k)
                temp += nn->d[i][k] * nn->weight[i][j][k];
            nn->d[i-1][j] = temp * nn->node[i][j] * (1 - nn->node[i][j]);
        }
    }
    //Show_nn_d(nn);
    //update weight
    for(i = 0; i < nn->layer-1; ++i)
        for(j = 0; j <= nn->node_num[i]; ++j)
            for(k = 0; k < nn->node_num[i+1]; ++k)
                nn->weight[i][j][k] += learning_rate * nn->d[i][k] * nn->node[i][j];

}

void update_input_layer(NN* nn, int step, int sample_n, int input_sample_dim, double* input)
{
    int i;
    for(i = 0; i < nn->node_num[0]; ++i)
        nn->node[0][i] = input[(step%sample_n)*input_sample_dim+i];
}

double cal_loss(NN* nn, int step, int sample_n, int output_sample_dim, double* output)
{
    double Loss = 0.0;
    int i;
    for(i = 0; i < nn->node_num[nn->layer-1]; ++i)
        Loss += (output[step%sample_n * output_sample_dim + i] - nn->node[nn->layer-1][i]) * (output[step%sample_n * output_sample_dim + i] - nn->node[nn->layer-1][i]);
    //printf("Loss = %f\n", Loss);
    return Loss;
}

double train(NN* nn, int step_num, double learning_rate, double* input, double* output, int sample_n, int input_sample_dim, int output_sample_dim, double er)
{
    int step = 0;
    double Loss_total = 1, Loss, Loss_temp = 0;
    Init_nn(nn, 0);
    //printf("%f", input[1]);
    while(step < step_num && Loss_total > er)
    {
        Init_nn(nn, 1);
        update_input_layer(nn, step, sample_n, input_sample_dim, input);
        //Show_nn(nn);
        foward_propagation(nn);
        Loss = cal_loss(nn, step, sample_n, output_sample_dim, output);
        //Show_nn(nn);
        back_propagation(nn, step, sample_n, output_sample_dim, output, learning_rate);
        //Show_nn(nn);
        Show_result(nn);
        Loss_temp += Loss;
        if(step%sample_n == 0 && step != 0)
        {
            Loss_total = Loss_temp;
            Loss_temp = 0;
            printf("Loss = %f\n", Loss_total);
            Save(Loss_total);
        }
        ++step;
    }
    printf("step = %d", step);
}

void test(NN* nn, double* input, double* output, int test_n, int input_sample_dim, int output_sample_dim)
{
    double Loss = 0;
    int i;
    printf("\n");
    for(i = 0; i < test_n; ++i)
    {
        update_input_layer(nn, i, test_n, input_sample_dim, input);
        foward_propagation(nn);
        Loss = cal_loss(nn, i, test_n, output_sample_dim, output);
        printf("%d", i);
        Show_result(nn);
        printf("Test_set[%d]: Loss = %f", i, Loss);
    }
}
