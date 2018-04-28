#ifndef _READFILE_H_
#define _READFILE_H_

#include <stdio.h>
void Save(double Loss);
void readfile(int a[], int b[], int m, int n);
void divide_data(int a[], int b[], double d_train_in[], double d_train_out[], double d_test_in[], double d_test_out[], int n_total, int n_dim, int n_train);
#endif // _READFILE_H_

