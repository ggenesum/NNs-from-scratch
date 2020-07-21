#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <err.h>
#include <time.h>
#include <string.h>

/*
This program learns the sequence "sesa" using a Recurrent Neural Network.
It is implemented using matrices.
*/

/* parameters */
#define InSize 5 //H(charset) alias (in this char2char configuration)
#define OutSize 5 //same
#define HiddenSize 25
#define epochs 5000
#define lr 2
#define seq_len 4
#define test_len 1

//macros for matrix dimensions
#define H(a) (sizeof(a) / sizeof(a[0]))
#define W(a) (sizeof(a[0]) / sizeof(a[0][0]))

char *seq[4] = { "s", "e", "s", "a" };
char *charset[5] = { "z", "t", "s", "e", "a" };

void Vectorize(int h, int w, long double m[h][w], long double (*f)(long double), long double output[h][w])
{ //apply function pointed by f to all elements of matrix m
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      output[i][j] = (*f)(m[i][j]);
    }
  }
}

long double sigmoid(long double i)
{
  long double z;
  if (i >=0)
  {
    return 1/(1+exp(-i));
  }
  else
  {
    z = exp(i);
    return z/(1+z);
  }
}

long double dsigmoid(long double i)
{
  long double s = sigmoid(i);
  return s*(1-s);
}

void add(int h1, int w1, long double m1[h1][w1], int h2, int w2, long double m2[h2][w2], long double output[h1][w1])
{ //add m1 and m2
  if (h1!=h2||w1!=w2)
  {
    printf("%ix%i + %ix%i = ?x?\n",h1,w1,h2,w2);
    errx(1, "cannot add matrices of different dimensions");
  }
  for (int i = 0; i < h1; i++)
  {
    for (int j = 0; j < w1; j++)
    {
      output[i][j] = m1[i][j] + m2[i][j];
    }
  }
}

void sub(int h1, int w1, long double m1[h1][w1], int h2, int w2, long double m2[h2][w2], long double output[h1][w1])
{ //output = m1 - m2
  if (h1!=h2||w1!=w2)
  {
    printf("%ix%i - %ix%i = ?x?\nv",h1,w1,h2,w2);
    errx(1, "cannot substract matrices of different dimensions");
  }
  for (int i = 0; i < h1; i++)
  {
    for (int j = 0; j < w1; j++)
    {
      output[i][j] = m1[i][j] - m2[i][j];
    }
  }
}

void EwiseDot(int h1, int w1, long double m1[h1][w1], int h2, int w2, long double m2[h2][w2], long double output[h1][w1])
{ //output = m1 * m2 (element wise)
  if (h1!=h2||w1!=w2)
  {
    printf("%ix%i - %ix%i = ?x?\nv",h1,w1,h2,w2);
    errx(1, "cannot multiply (element wise) matrices of different dimensions");
  }
  for (int i = 0; i < h1; i++)
  {
    for (int j = 0; j < w1; j++)
    {
      output[h1][w1] = m1[i][j] * m2[i][j];
    }
  }
}

void dot(int h1, int w1, long double m1[h1][w1], int h2, int w2, long double m2[h2][w2], long double output[h1][w2])
{ //m1*m2
  long double tmp; //use temp var to allow saving in m1 or m2

  if (w1 != h2)
  {
    printf("%ix%i * %ix%i = ?x?\n",h1,w1,h2,w2);
    errx(1, "cannot multiply matrices if dim 1 (m1) != dim 0 (m2)");
  }
  for (int i = 0; i < h1; i++)
  {
    for (int j = 0; j < w2; j++)
    {
      tmp = 0;
      for (int k = 0; k < w1; k++)
      {
        tmp += m1[i][k] * m2[k][j];
      }
      output[i][j] = tmp;
    }
  }
}

void scalar_dot(int h, int w, long double m[h][w], long double n, long double output[h][w])
{ //n*M, n scalar and M 2D array
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      output[i][j] = n*m[i][j];
    }
  }
}

void Softmax(int h, long double m[h][1], long double output[h][1])
{
long double max = -INFINITY;
 for (size_t i = 0; i < h; i++) {
   if (m[i][0] > max) {
     max = m[i][0];
   }
 }

 long double sum = 0.0;
 for (size_t i = 0; i < h; i++) {
   sum += expf(m[i][0] - max);
 }

 float offset = max + logf(sum);
 for (size_t i = 0; i < h; i++) {
   output[i][0] = expf(m[i][0] - offset);
 }
}

long double rnd(long double i)
{
    return (rand()/(float)RAND_MAX*2-1)*0.04;
}

long double clip(long double i)
{ //clip to -1, 1
  if (i<-1)
  {
    return (float)-1;
  }
  else if (i>1)
  {
    return (float)1;
  }
  else
  {
    return i;
  }
}

void print(int h, int w, long double m[h][w])
{
  //int c = 0;
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
        printf("%Lf - ",m[i][j]);
    }
    printf("\n");
  }
  printf("\n");
}

void char2vect(char c,long double output[OutSize][1])
{
  for (int i = 0; i < OutSize; i++)
  {
    if (charset[i][0] == c)
    {
      output[i][0] = 1;
    }
    else
    {
      output[i][0] = 0;
    }
  }
}


void Transpose(int h, int w, long double m[h][w], long double output[w][h])
{
  for (int i = 0; i < w; i++)
  {
    for (int j = 0; j < h; j++)
    {
      output[i][j] = m[j][i];
    }
  }
}

void Copy(int h, int w, long double m[h][w], long double output[h][w])
{
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      output[i][j] = m[i][j];
    }
  }
}

int main()
{
  long double Wh[HiddenSize][HiddenSize] = {{0}};
  long double Wx[HiddenSize][InSize] = {{0}};
  long double Ba[HiddenSize][1] = {{0}};
  long double Wy[OutSize][HiddenSize] = {{0}};
  long double By[OutSize][1] = {{0}};
  long double _Y[OutSize][1] = {{0}};
  long double _A[HiddenSize][1] = {{0}};
  long double _H[HiddenSize][1] = {{0}};

  long double LossError[OutSize][1] = {{0}}; //gradient of the lost with respect to O, with Loss = Cross_Enthropy(Softmax(_O)) with _Y = Softmax(_O)
  long double WhUpdate[HiddenSize][HiddenSize] = {{0}}; //gradient of the loss with repect to _H
  long double WxUpdate[HiddenSize][InSize] = {{0}};
  long double BaUpdate[HiddenSize][1] = {{0}};
  long double WyUpdate[OutSize][HiddenSize] = {{0}};
  long double ByUpdate[OutSize][1] = {{0}};

  long double Input[OutSize][1] = {{0}}; //Input data at timestep t
  long double Target[OutSize][1] = {{0}}; //Target data (next Input)
  char timesteps[seq_len-1][2] = {{0}};

  long double LossErrors[seq_len-1][OutSize][1] = {{{0}}};  //loss at each timestep
  long double Att[seq_len][H(_A)][W(_A)] = {{{0}}}; //Att stores _A at each timestep
  long double Htt[seq_len-1][H(_H)][W(_H)] = {{{0}}}; //same for _H
  long double Inputs[seq_len-1][OutSize][1] = {{{0}}};
  long double Targets[seq_len-1][OutSize][1]= {{{0}}};

  long double Wx_X[HiddenSize][1] = {{0}}; //temporary variables for feedforward calculus
  long double Wh_H[HiddenSize][1] = {{0}}; // = Wh*_Ht-1

  long double dL_dh[HiddenSize][1] = {{0}}; //temporary variables for backpropagation trough time (BPTT) calculus
  long double temp[HiddenSize][1] = {{0}}; //dL_dH(t) * dH(t)/dH(t-1) : this recurrence relation is used to backpropagate error of H trough previous timesteps
  long double WyT[HiddenSize][OutSize] = {{0}}; //Transposed Wy
  long double HtT[W(_H)][H(_H)] = {{0}};
  long double AtT[W(_A)][H(_A)] = {{0}};
  long double ItT[1][OutSize] = {{0}};
  long double WyUpdTemp[OutSize][HiddenSize] = {{0}}; //temp matrix for WyUpdate calculus
  long double WhUpdTemp[HiddenSize][HiddenSize] = {{0}};
  long double WxUpdTemp[HiddenSize][InSize] = {{0}};

  long double max;
  int max_i;

  srand ( time(NULL) ); //seed random generator
  //random weights, biases and output let to zero
  Vectorize(H(Wh),W(Wh),Wh,rnd,Wh); //apply function rnd to all elements of Wh and save the result directly into Wh
  Vectorize(H(Wx),W(Wx),Wx,rnd,Wx);
  Vectorize(H(Wy),W(Wy),Wy,rnd,Wy);

  for (int epoch = 0; epoch < epochs; epoch++)
  {
    //generate test data
    //"sesa" sequence over and over again

    for (int i = 0; i < seq_len-1; i++)
    {
      timesteps[i][0] = *seq[i];
      timesteps[i][1] = *seq[i+1];
      //ex timestep 0 : Input t, Target e
    }

    memset(Att, 0, sizeof(Att));
    memset(_H, 0, sizeof(_H)); //reset memory of the rnn
    memset(LossError, 0, sizeof(LossError));
    memset(WhUpdate, 0, sizeof(WhUpdate));
    memset(WxUpdate, 0, sizeof(WxUpdate));
    memset(BaUpdate, 0, sizeof(BaUpdate));
    memset(WyUpdate, 0, sizeof(WyUpdate));
    memset(ByUpdate, 0, sizeof(ByUpdate));

    //feedforward
    for (int t = 0; t < seq_len-1; t++)
    {
      //encode character to one-hot vector
      char2vect(timesteps[t][0],Input);
      char2vect(timesteps[t][1],Target);

      //save all inputs and outputs
      Copy(H(Input),W(Input), Input, Inputs[t]);
      Copy(H(Target),W(Target), Target, Targets[t]);

      //feedforward
      dot(H(Wx), W(Wx), Wx, H(Input), W(Input), Input, Wx_X); //Wx_X = Wx*_X
      dot(H(Wh), W(Wh), Wh, H(_H), W(_H), _H, Wh_H); //Wh_H = Wh*_H
      add(H(Wh_H), W(Wh_H), Wh_H, H(Wx_X), W(Wx_X), Wx_X, _A); //_A = Wh*_H(t-1) + Wx*_X
      add(H(_A), W(_A), _A, H(Ba), W(Ba), Ba, _A); //_A += Ba
      Vectorize(H(_A), W(_A), _A, sigmoid, _H); //_H = Sigmoid(_A)
      dot(H(Wy), W(Wy), Wy, H(_H), W(_H), _H, _Y); //_Y = Wy*_H(t)
      add(H(_Y), W(_Y), _Y, H(By), W(By), By, _Y); //_Y += By
      Softmax(H(_Y), _Y, _Y); //_Y = Softmax(_Y)

      sub(H(_Y), W(_Y), _Y, H(Target), W(Target), Target, LossError); //loss derivative (cross entrophy + softmax)

      //save value of _A, _H and LossError for BPTT
      Copy(H(_A),W(_A), _A, Att[t+1]); //t+1 because we have one more element here, A(t=0)
      Copy(H(_H),W(_H), _H, Htt[t]);
      Copy(H(LossError),W(LossError), LossError, LossErrors[t]);
    }

    //backpropagation
    for (int j = 0; j < H(LossErrors); j++)
    {
      Transpose(OutSize, HiddenSize, Wy, WyT); //transpose Wy in WyT
      dot(H(WyT), W(WyT), WyT, H(LossErrors[j]), W(LossErrors[j]), LossErrors[j], dL_dh); //dL/dH
      Transpose(H(Htt[j]), W(Htt[j]), Htt[j], HtT);
      dot(H(LossErrors[j]), W(LossErrors[j]), LossErrors[j], H(HtT),  W(HtT), HtT, WyUpdTemp);
      add(H(WyUpdTemp), W(WyUpdTemp), WyUpdTemp, H(WyUpdate), W(WyUpdate), WyUpdate, WyUpdate); //dL/dWy, we sum update values at each timestep
      add(H(ByUpdate), W(ByUpdate), ByUpdate, H(LossErrors[j]), W(LossErrors[j]), LossErrors[j], ByUpdate);

      for (int k = j-1; k > -1; k--)
      { /*
        dLoss/dWh(t),dLoss/dWx(t) and dLoss/dBa(t) depends of previous states as well
        (H(t-1) is in H(t))
        - the network can learn trough time because of this part -
        */
        Vectorize(H(Att[k+1]), W(Att[k+1]), Att[k+1], dsigmoid, Att[k+1]); //save Dsigmoid(Att[k+1]) in Att[k+1]; we can do that becaus we never use Att[k+1] again
        EwiseDot(H(Att[k+1]), W(Att[k+1]), Att[k+1], H(dL_dh), W(dL_dh), dL_dh, temp);

        Transpose(H(Att[k]), W(Att[k]), Att[k], AtT); //transpose Wy in WyT
        dot(H(temp), W(temp), temp, H(AtT), W(AtT), AtT, WhUpdTemp); //Wh update for this timestep
        add(H(WhUpdTemp), W(WhUpdTemp), WhUpdTemp, H(WhUpdate), W(WhUpdate), WhUpdate, WhUpdate); //sum

        Transpose(H(Inputs[k]), W(Inputs[k]), Inputs[k], ItT);
        dot(H(temp), W(temp), temp, H(ItT), W(ItT), ItT, WxUpdTemp);
        add(H(WxUpdTemp), W(WxUpdTemp), WxUpdTemp, H(WxUpdate), W(WxUpdate), WxUpdate, WxUpdate);

        add(H(BaUpdate), W(BaUpdate), BaUpdate, H(temp), W(temp), temp, BaUpdate);

        dot(H(Wh), W(Wh), Wh, H(temp), W(temp), temp, dL_dh); //temp*=dH(t)/dH(t-1)
      }

      //clip gradients
      Vectorize(H(WhUpdate), W(WhUpdate), WhUpdate, clip, WhUpdate);
      Vectorize(H(WxUpdate), W(WxUpdate), WxUpdate, clip, WxUpdate);
      Vectorize(H(BaUpdate), W(BaUpdate), BaUpdate, clip, BaUpdate);
      Vectorize(H(WyUpdate), W(WyUpdate), WyUpdate, clip, WyUpdate);
      Vectorize(H(ByUpdate), W(ByUpdate), ByUpdate, clip, ByUpdate);

      //SGD
      //scale by learning rate
      scalar_dot(H(WhUpdate), W(WhUpdate), WhUpdate, lr, WhUpdate);
      scalar_dot(H(WxUpdate), W(WxUpdate), WxUpdate, lr, WxUpdate);
      scalar_dot(H(BaUpdate), W(BaUpdate), BaUpdate, lr, BaUpdate);
      scalar_dot(H(WyUpdate), W(WyUpdate), WyUpdate, lr, WyUpdate);
      scalar_dot(H(ByUpdate), W(ByUpdate), ByUpdate, lr, ByUpdate);

      //update weights
      sub(H(Wh), W(Wh), Wh, H(WhUpdate), W(WhUpdate), WhUpdate, Wh);
      sub(H(Wx), W(Wx), Wh, H(WxUpdate), W(WxUpdate), WxUpdate, Wx);
      sub(H(Ba), W(Ba), Wh, H(BaUpdate), W(BaUpdate), BaUpdate, Ba);
      sub(H(Wy), W(Wy), Wy, H(WyUpdate), W(WyUpdate), WyUpdate, Wy);
      sub(H(By), W(By), By, H(ByUpdate), W(ByUpdate), ByUpdate, By);

    }
  }

  //test
  for (int p = 0; p < test_len; p++)
  {
    memset(_H, 0, sizeof(_H)); //reset memory of the rnn
    printf("%c",timesteps[0][0]);
    for (int t = 0; t < seq_len-1; t++)
    {
      //encode character to one-hot vector
      char2vect(timesteps[t][0],Input);
      char2vect(timesteps[t][1],Target);

      //feedforward
      dot(H(Wx), W(Wx), Wx, H(Input), W(Input), Input, Wx_X); //Wx_X = Wx*_X
      dot(H(Wh), W(Wh), Wh, H(_H), W(_H), _H, Wh_H); //Wh_H = Wh*_H
      add(H(Wh_H), W(Wh_H), Wh_H, H(Wx_X), W(Wx_X), Wx_X, _A); //_A = Wh*_H(t-1) + Wx*_X
      add(H(_A), W(_A), _A, H(Ba), W(Ba), Ba, _A); //_A += Ba
      Vectorize(H(_A), W(_A), _A, sigmoid, _H); //_H = Sigmoid(_A)
      dot(H(Wy), W(Wy), Wy, H(_H), W(_H), _H, _Y); //_Y = Wy*_H(t)
      add(H(_Y), W(_Y), _Y, H(By), W(By), By, _Y); //_Y += By
      Softmax(H(_Y), _Y, _Y); //_Y = Softmax(_Y)


      max = 0;
      max_i = 0;
      for (int k = 0; k < H(_Y); k++)
      {
        if (_Y[k][0] > max)
        {
          max = _Y[k][0];
          max_i = k;
        }
      }
      printf("%s",charset[max_i]);
    }

  }
  return 0;
}
