%module DeconwareSwig
%include arrays_java.i
%apply float[] {float *};
%{
extern float dot_device(size_t N, float *in1, float *in2);
%}

extern float dot_device(size_t N, float *in1, float *in2);

