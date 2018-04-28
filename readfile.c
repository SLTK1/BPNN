#include "readfile.h"

void readfile(int a[], int b[], int m, int n)
{
    int i;
    FILE *fp1, *fp2;
    fp1 = fopen("parity_code.txt", "rb");
    if(fp1 == NULL)
    {
        printf("error");
        return;
    }
    fread(a , sizeof(int), m*n, fp1);
    fp2 = fopen("parity_code_label.txt", "rb");
    if(fp2 == NULL)
    {
        printf("error");
        return;
    }
    fread(b , sizeof(int), m*n, fp2);
    fclose(fp1);
    fclose(fp2);
    /*
    for(i = 0; i < m*n; ++i)
    {
        if(i%7 == 0)
            printf("\n");
        printf("%d", a[i]);
    }
    */
}

void divide_data(int a[], int b[], double d_train_in[], double d_train_out[], double d_test_in[], double d_test_out[], int n_total, int n_dim, int n_train)
{
    int i;
    for(i = 0; i < n_train * n_dim; ++i)
        d_train_in[i] = a[i];
    for(i = n_train * n_dim; i < n_total * n_dim; ++i)
        d_test_in[i] = a[i];
    for(i = 0; i < n_train; ++i)
        d_train_out[i] = b[i];
    for(i = n_train; i < n_total; ++i)
        d_test_out[i] = b[i];
    /*
    for(int i = 0; i < n_train * n_dim; ++i)
    {
        if(i % 7 == 0)
            printf("\n");
        printf("%f", d_train_in[i]);
    }
    */
}

void Save(double Loss)
{
    FILE *fp;
    fp = fopen("loss.bin","a");
    if(fp == 0)
    {
        printf("error\n");
        return;
    }
    //fseek(fp, 0, SEEK_END);
    fwrite(&Loss, sizeof(double), 1, fp);
    fclose(fp);
}
