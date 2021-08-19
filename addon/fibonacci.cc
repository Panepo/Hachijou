#include <napi.h>

int factorialize(int num){
  int result = num;
  if( num == 0 || num == 1 ) return 1;
  while ( num > 1) {
    num--;
    result *= num;
  }
  return result;
}

int factorializeFib(int num) {
  int a = 1, b = 0, temp = 1;

  while (num >= 0) {
    temp = a;
    a = a + b;
    b = temp;
    factorialize(b);
    num--;
  }

  return b;
}

Napi::Value factorializeWrapper(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1) {
    Napi::TypeError::New(env, "Wrong number of arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  double arg0 = info[0].As<Napi::Number>().DoubleValue();
  Napi::Number num = Napi::Number::New(env, factorialize(arg0));

  return num;
}

Napi::Value factorializeFibWrapper(const Napi::CallbackInfo& info) {
  Napi::Env env = info.Env();

  if (info.Length() < 1) {
    Napi::TypeError::New(env, "Wrong number of arguments")
        .ThrowAsJavaScriptException();
    return env.Null();
  }

  if (!info[0].IsNumber()) {
    Napi::TypeError::New(env, "Wrong arguments").ThrowAsJavaScriptException();
    return env.Null();
  }

  double arg0 = info[0].As<Napi::Number>().DoubleValue();
  Napi::Number num = Napi::Number::New(env, factorializeFib(arg0));

  return num;
}

Napi::Object Init(Napi::Env env, Napi::Object exports) {
  exports.Set(Napi::String::New(env, "factorialize"), Napi::Function::New(env, factorializeWrapper));
  exports.Set(Napi::String::New(env, "factorializeFib"), Napi::Function::New(env, factorializeFibWrapper));
  return exports;
}

NODE_API_MODULE(addon, Init)
