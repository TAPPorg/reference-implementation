/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include "tblis_bind.h"
#include "mpi_utils.h"
#include <chrono>
#include <thread>
#include <optional>
// using namespace CTF;

namespace {

void run_tblis_mult_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, float* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, float* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, float* D, int op_D, int64_t* idx_D,
                    float alpha, float beta);
bool compare_tensors_s(float* A, float* B, int size);

void run_tblis_mult_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, double* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, double* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, double* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, double* D, int op_D, int64_t* idx_D,
                    double alpha, double beta);
bool compare_tensors_d(double* A, double* B, int size);

void run_tblis_mult_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<float>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<float>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<float>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<float>* D, int op_D, int64_t* idx_D,
                    std::complex<float> alpha, std::complex<float> beta);
bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size);

void run_tblis_mult_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<double>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<double>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<double>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<double>* D, int op_D, int64_t* idx_D,
                    std::complex<double> alpha, std::complex<double> beta);
bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size);


std::string str(bool b);
void execute_product_tblis_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);
void execute_product_tblis_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);
void execute_product_tblis_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);
void execute_product_tblis_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta);



int initialize(){
  int rank, np, n, pass;
  int  in_num = 3;
  char ** input_str = new char*[in_num];;

  input_str[0] = "test++";
  input_str[1] = "-n";
  input_str[2] = "1000";
  n = 1000;
  int initialized, finalized;

  std::cout << "mpi1" << std::endl;
  MPI_Initialized(&initialized);

  std::cout << "mpi2" << std::endl;
  if (!initialized) {
    std::cout << "initializing mpi" << std::endl;
  	//MPI_Init(&in_num, &input_str);
  	MPI_Init(nullptr, nullptr);

    MPI_Group world_group, new_group;
    MPI_Comm new_comm;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    int ranks_to_exclude[] = {0}; // Exclude ranks 0 and 2
    int num_ranks_to_exclude = 1;

    MPI_Group_excl(world_group, num_ranks_to_exclude, ranks_to_exclude, &new_group);

    MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);
    {
      CTF::World dw0(in_num, input_str);
    }
  //MPI_Group_free(&new_group);
  }
  return 0;
}



std::string generateUUID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    int i;
    ss << std::hex;
    for (i = 0; i < 16; i++) {
        if (i == 4 || i == 6 || i == 8 || i == 10) {
            ss << '-';
        }
        int v;
        if (i == 6) {
            v = 4;
        } else if (i == 8) {
            v = dis2(gen);
        } else {
            v = dis(gen);
        }
        ss << v;
    }
    return ss.str();
}



int doJacobi(){

  std::string message = "doJacobi";
  mpiBroadcastString(message);

  waitWorkersFinished();

  return 0;
}

int finalize(){

  std::string message = "stopWorker";
  mpiBroadcastString(message);
  waitWorkersFinished();

  int finalized;
  MPI_Finalized(&finalized);
  if(!finalized){
    MPI_Finalize();
  }
  // MPI_Group_free(&new_group);
  // MPI_Comm_free(&new_comm);
  return 0;
}


void run_tblis_mult_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, float* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, float* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, float* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, float* D, int op_D, int64_t* idx_D,
                    float alpha, float beta)
{
  std::cout << " test++ run_tblis_mult_s does nothing" << std::endl; 
  exit(-1);
}


void run_tblis_mult_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, double* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, double* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, double* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, double* D, int op_D, int64_t* idx_D,
                    double alpha, double beta)
{
}

void run_tblis_mult_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<float>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<float>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<float>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<float>* D, int op_D, int64_t* idx_D,
                    std::complex<float> alpha, std::complex<float> beta)
{
}


void run_tblis_mult_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, std::complex<double>* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, std::complex<double>* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, std::complex<double>* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, std::complex<double>* D, int op_D, int64_t* idx_D,
                    std::complex<double> alpha, std::complex<double> beta)
{
  std::cout << " run_tblis_mult_z" << std::endl; 
  initialize();
  int datatype = TAPP_C64;
  std::string uuid_A = generateUUID();
  std::string name = "A";
  std::complex<double> init_val = 0.0;
  createTensor(uuid_A, nmode_A, extents_A, datatype, name, init_val);
  distributeArrayDataPart1(uuid_A);
  distributeArrayDataPart2(A, 0, nullptr, nullptr, datatype);
  int64_t numel = 1;
  for(int i=0;i<nmode_A;i++) numel *= extents_A[i];
  for(int i=0;i<numel;i++) std::cout << " A " << A[i] << std::endl;

  finalize();
  exit(0);
}



bool compare_tensors_s(float* A, float* B, int size)
{
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        float rel_diff = abs((A[i] - B[i]) / (A[i] > B[i] ? A[i] : B[i]));
        if (rel_diff > 0.00005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << rel_diff << std::endl;
            found = true;
        }
    }
    return !found;
}

bool compare_tensors_d(double* A, double* B, int size)
{
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        double rel_diff = abs((A[i] - B[i]) / (A[i] > B[i] ? A[i] : B[i]));
        if (rel_diff > 0.00005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << rel_diff << std::endl;
            found = true;
        }
    }
    return !found;
}

bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size)
{
    bool found = false;
    for (int i = 0; i < size; i++)
    {
        float rel_diff_r = abs((A[i].real() - B[i].real()) / (A[i].real() > B[i].real() ? A[i].real() : B[i].real()));
        float rel_diff_i = abs((A[i].imag() - B[i].imag()) / (A[i].imag() > B[i].imag() ? A[i].imag() : B[i].imag()));
        if (rel_diff_r > 0.00005 || rel_diff_i > 0.00005)
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << std::complex<float>(rel_diff_r, rel_diff_i) << std::endl;
            found = true;
        }
    }
    return !found;
}

bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size_)
{
    bool found = false;
    for (int i = 0; i < size_; i++)
    {
        double rel_diff_r = abs((A[i].real() - B[i].real()) / (A[i].real() > B[i].real() ? A[i].real() : B[i].real()));
        double rel_diff_i = abs((A[i].imag() - B[i].imag()) / (A[i].imag() > B[i].imag() ? A[i].imag() : B[i].imag()));
        double abs_diff_r = abs(A[i].real() - B[i].real());
        double abs_diff_i = abs(A[i].imag() - B[i].imag());
        if ((rel_diff_r > 0.00005 || rel_diff_i > 0.00005) && (abs_diff_r > 1e-12 || abs_diff_i >  1e-12))
        {
            std::cout << "\n" << i << ": " << A[i] << " - " << B[i] << std::endl;
            std::cout << "\n" << i << ": " << std::complex<double>(rel_diff_r, rel_diff_i) << std::endl;
            std::cout << "\n" << " size: " << size_ << ". " << std::endl;
            found = true;
        }
    }
    return !found;
}



std::string str(bool b)
{
    return b ? "true" : "false";
}

void execute_product_tblis_s(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  float* A_ = static_cast<float*>(A);
  float* B_ = static_cast<float*>(B);
  float* C_ = static_cast<float*>(C);
  float* D_ = static_cast<float*>(D);
  float* alpha_ = static_cast<float*>(alpha);
  float* beta_ = static_cast<float*>(beta);

  run_tblis_mult_s(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}

void execute_product_tblis_d(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  double* A_ = static_cast<double*>(A);
  double* B_ = static_cast<double*>(B);
  double* C_ = static_cast<double*>(C);
  double* D_ = static_cast<double*>(D);
  double* alpha_ = static_cast<double*>(alpha);
  double* beta_ = static_cast<double*>(beta);

  run_tblis_mult_d(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}

void execute_product_tblis_c(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  std::complex<float>* A_ = static_cast<std::complex<float>*>(A);
  std::complex<float>* B_ = static_cast<std::complex<float>*>(B);
  std::complex<float>* C_ = static_cast<std::complex<float>*>(C);
  std::complex<float>* D_ = static_cast<std::complex<float>*>(D);
  std::complex<float>* alpha_ = static_cast<std::complex<float>*>(alpha);
  std::complex<float>* beta_ = static_cast<std::complex<float>*>(beta);

  run_tblis_mult_c(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}

void execute_product_tblis_z(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                  int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                  int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                  int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                  void* alpha, void* beta)
{
  std::complex<double>* A_ = static_cast<std::complex<double>*>(A);
  std::complex<double>* B_ = static_cast<std::complex<double>*>(B);
  std::complex<double>* C_ = static_cast<std::complex<double>*>(C);
  std::complex<double>* D_ = static_cast<std::complex<double>*>(D);
  std::complex<double>* alpha_ = static_cast<std::complex<double>*>(alpha);
  std::complex<double>* beta_ = static_cast<std::complex<double>*>(beta);

  run_tblis_mult_z(nmode_A, extents_A, strides_A, A_, op_A, idx_A,
                   nmode_B, extents_B, strides_B, B_, op_B, idx_B,
                   nmode_C, extents_C, strides_C, C_, op_C, idx_C,
                   nmode_D, extents_D, strides_D, D_, op_D, idx_D,
                   *alpha_, *beta_);
}


}

extern "C" {
  void bind_tblis_execute_product(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                    void* alpha, void* beta, int datatype_tapp){
    switch (datatype_tapp) {
      case TAPP_F32:
        execute_product_tblis_s(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
      case TAPP_F64:
        execute_product_tblis_d(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
      case TAPP_C32:
        execute_product_tblis_c(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
      case TAPP_C64:
        execute_product_tblis_z(nmode_A, extents_A, strides_A, A, op_A, idx_A, nmode_B, extents_B, strides_B, B, op_B, idx_B,
                         nmode_C, extents_C, strides_C, C, op_C, idx_C, nmode_D, extents_D, strides_D, D, op_D, idx_D,
                         alpha, beta);
        break;
    } 
  }

int compare_tensors_(void* A, void* B, int64_t size, int datatype_tapp){
    bool result = false;
    switch (datatype_tapp) { // tapp_datatype
      case TAPP_F32:
        {
          float* A_ = static_cast<float*>(A);
          float* B_ = static_cast<float*>(B);
          result = compare_tensors_s(A_, B_, (int)size);
        }
        break;
      case TAPP_F64:
        {
          double* A_ = static_cast<double*>(A);
          double* B_ = static_cast<double*>(B);
          result = compare_tensors_d(A_, B_, (int)size);
        }
        break;
      case TAPP_C32:
        {
          std::complex<float>* A_ = static_cast<std::complex<float>*>(A);
          std::complex<float>* B_ = static_cast<std::complex<float>*>(B);
          result = compare_tensors_c(A_, B_, (int)size);
        }
        break;
      case TAPP_C64:
        {
          std::complex<double>* A_ = static_cast<std::complex<double>*>(A);
          std::complex<double>* B_ = static_cast<std::complex<double>*>(B);
          result = compare_tensors_z(A_, B_, (int)size);
        }
        break;
    } 
    return result; 
  }









}


