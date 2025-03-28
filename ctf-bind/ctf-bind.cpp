/*
 * Niklas Hörnblad
 * Paolo Bientinesi
 * Umeå University - June 2024
 */

#include "ctf-bind.h"
#include "mpi_utils.h"
#include <chrono>
#include <thread>
#include <optional>
// using namespace CTF;

namespace {

bool compare_tensors_s(float* A, float* B, int size);

bool compare_tensors_d(double* A, double* B, int size);

bool compare_tensors_c(std::complex<float>* A, std::complex<float>* B, int size);

bool compare_tensors_z(std::complex<double>* A, std::complex<double>* B, int size);

//implementation
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

} // end of namespace

int initialize(){
  int rank, np, n, pass;
  int  in_num = 3;
  char ** input_str = new char*[in_num];;

  input_str[0] = "test++";
  input_str[1] = "-n";
  input_str[2] = "1000";
  n = 1000;
  int initialized, finalized;

  MPI_Initialized(&initialized);

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
  exit(0);
  return 0;
}


extern "C" {

  int finalizeWork(){
    return finalize();
  }

  void ctf_bind_execute_product(int nmode_A, int64_t* extents_A, int64_t* strides_A, void* A, int op_A, int64_t* idx_A,
                    int nmode_B, int64_t* extents_B, int64_t* strides_B, void* B, int op_B, int64_t* idx_B,
                    int nmode_C, int64_t* extents_C, int64_t* strides_C, void* C, int op_C, int64_t* idx_C,
                    int nmode_D, int64_t* extents_D, int64_t* strides_D, void* D, int op_D, int64_t* idx_D,
                    void* alpha, void* beta, int datatype_tapp){

    switch (datatype_tapp) {
      case TAPP_F32:
        std::cout << "Error: contraction with TAPP_F32 not implemented with Cyclops CTF. " <<  std::endl;
        exit(489);
        break;
      case TAPP_F64:
        //ok
        break;
      case TAPP_C32:
        std::cout << "Error: contraction with TAPP_C32 not implemented with Cyclops CTF. " <<  std::endl;
        exit(489);
        break;
      case TAPP_C64:
        //ok
        break;
    } 

  initialize();
  std::complex<double> init_val = 0.0;

  std::string uuid_A = generateUUID();
  std::string name_A = "A";
  createTensor(uuid_A, nmode_A, extents_A, datatype_tapp, name_A, init_val);
  distributeArrayDataPart1(uuid_A);
  distributeArrayDataPart2(A, 0, nullptr, nullptr, datatype_tapp);

  std::string uuid_B = generateUUID();
  std::string name_B = "B";
  createTensor(uuid_B, nmode_B, extents_B, datatype_tapp, name_B, init_val);
  distributeArrayDataPart1(uuid_B);
  distributeArrayDataPart2(B, 0, nullptr, nullptr, datatype_tapp);

  std::string uuid_C = generateUUID();
  std::string name_C = "C";
  createTensor(uuid_C, nmode_C, extents_C, datatype_tapp, name_C, init_val);
  distributeArrayDataPart1(uuid_C);
  distributeArrayDataPart2(C, 0, nullptr, nullptr, datatype_tapp);

  std::string uuid_D;
  if (C == D) uuid_D = uuid_C;
  else {
    uuid_D = generateUUID();
 
    std::string name_D = "D";
    createTensor(uuid_D, nmode_D, extents_D, datatype_tapp, name_D, init_val);
  }

  std::complex<double> alpha_;
  std::complex<double> beta_;

  if(datatype_tapp == TAPP_F64){
    alpha_ = std::complex<double>(*static_cast<double*>(alpha));
    beta_ = std::complex<double>(*static_cast<double*>(beta));
  }
  else if(datatype_tapp == TAPP_C64){
    alpha_ = *static_cast<std::complex<double>*>(alpha);
    beta_ = *static_cast<std::complex<double>*>(beta);
  }
  else {
        std::cout << "Error: contraction with the requested datatype not implemented with Cyclops CTF. " <<  std::endl;
        exit(489);
  }

  executeProduct(uuid_A, nmode_A, idx_A, uuid_B, nmode_B, idx_B, uuid_C, nmode_C, idx_C, uuid_D, nmode_D, idx_D, alpha_, beta_);

  gatherDistributedArrayDataPart1(uuid_D);
  gatherDistributedArrayDataPart2(D, 0, nullptr, nullptr, datatype_tapp);

  int64_t numel = 1;
  for(int i=0;i<nmode_D;i++) numel *= extents_D[i];
  for(int i=0;i<numel;i++) std::cout << " D " << (static_cast<std::complex<double>*>(D))[i] << std::endl;

  if(datatype_tapp == TAPP_C64){
    std:: string uuid_D2 = generateUUID();
    // for(int i=0;i<numel;i++) Dorig[i] = (static_cast<std::complex<double>*>(D))[i];
    std::string name_D2 = "D2";
    createTensor(uuid_D2, nmode_D, extents_D, datatype_tapp, name_D2, init_val);
    distributeArrayDataPart1(uuid_D2);
    distributeArrayDataPart2(D, 0, nullptr, nullptr, datatype_tapp);
    std::complex<double> * D2 = new std::complex<double>[numel];
    gatherDistributedArrayDataPart1(uuid_D2);
    gatherDistributedArrayDataPart2(D2, 0, nullptr, nullptr, datatype_tapp);
    for(int i=0;i<numel;i++) if (D2[i] != (static_cast<std::complex<double>*>(D))[i]) 
      std::cout << " Differ Dorig D2 " << (static_cast<std::complex<double>*>(D))[i] << " " << D2[i] << std::endl;
    
    destructTensor(uuid_D2);
    delete[] D2; 
  }
  std::cout << " op_D " << op_D << " op_A " << op_A << " op_B " << op_B << " op_C " << op_C << std::endl;

  destructTensor(uuid_A);
  destructTensor(uuid_B);
  destructTensor(uuid_C);
  if (uuid_C != uuid_D) destructTensor(uuid_D);
  // finalize();
  // exit(0);

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


