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
#include <cstring>
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

  //input_str[1] = "-n";
  //input_str[2] = "1000";
  int initialized;

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

		// create CTF::World
  	int  in_num = 1;
  	char ** input_str = new char*[in_num];
    input_str[0] = new char[1] {'\0'};

    CTF::World dw0(in_num, input_str);

		delete[] input_str[0];
		delete[] input_str;
  //MPI_Group_free(&new_group);
  }
  // delete input str
  return 0;
}


int doJacobi(){

  std::string message = "doJacobi";
  mpiBroadcastString(message);

  waitWorkersFinished();

  return 0;
}

int initialize_(){
  initialize(); 
  std::string message = "initializeWorker";
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

std::string generateUUID() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 15);
    std::uniform_int_distribution<> dis2(8, 11);

    std::stringstream ss;
    int i;
    ss << std::hex;
    int len = distributed_get_uuid_len() - 4;
    for (i = 0; i < len; i++) {
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


extern "C" {
 
  int distributed_get_uuid_len(){
    return 20;
  }



  int distributed_get_uuid(char * uuid, const int uuid_len){
    std::string uuid_ = generateUUID();
    int len = uuid_.length();
    if(uuid_len != len) {
      std::cout << "Error: insufficient char buffer size in distributed_get_uuid. " << uuid_len << " " << len << " " << uuid_<<  std::endl;
      exit(446);
    }
    std::strcpy(uuid, uuid_.c_str());
    return 0;
  }

  int distributed_initialize_tensor(TAPP_tensor_info info, void* init_val){
    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);

    TAPP_datatype datatype_tapp = TAPP_get_datatype(info);

    std::complex<double> init_val_ = 0.0;
    if(datatype_tapp == TAPP_F64){
      init_val_ = std::complex<double>(*static_cast<double*>(init_val));
    }
    else if(datatype_tapp == TAPP_C64){
      init_val_ = *static_cast<std::complex<double>*>(init_val);
    }
    else {
      std::cout << "Error: distributed_initialize_tensor with the requested datatype not implemented with Cyclops CTF. " <<  std::endl;
      exit(489);
    }
    
    initializeTensor(uuid, init_val_);

    delete[] uuid_;
    return 0;
  }

  int distributed_copy_tensor(TAPP_tensor_info dest, TAPP_tensor_info src){
    int uuid_len = TAPP_get_uuid_len(src); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr;

    ierr = TAPP_get_uuid(dest, uuid_, uuid_len);
    std::string uuid_dest(uuid_);
   
    ierr = TAPP_get_uuid(src, uuid_, uuid_len);
    std::string uuid_src(uuid_);

    copyTensor(uuid_dest, uuid_src);

    delete[] uuid_;
    return 0;
  }

  int distributed_reshape_tensor(TAPP_tensor_info dest, TAPP_tensor_info src, int nmode, int64_t* extents){
    int uuid_len = TAPP_get_uuid_len(src); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr;

    ierr = TAPP_get_uuid(dest, uuid_, uuid_len);
    std::string uuid_dest(uuid_);
   
    ierr = TAPP_get_uuid(src, uuid_, uuid_len);
    std::string uuid_src(uuid_);

    reshapeTensor(uuid_dest, uuid_src, nmode, extents);

    delete[] uuid_;
    return 0;
  }

  int distributed_scale_with_denominators(TAPP_tensor_info info, 
		                  const int n_occ, const int n_vir, void* eps_occ, void* eps_vir, void* eps_ijk){
    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);
    
    TAPP_datatype datatype_tapp = TAPP_get_datatype(info);
    int n_occ_ = n_occ;
    int n_vir_ = n_vir;

    scaleWithDenominatorsPart1(uuid, n_occ_, n_vir_);
    scaleWithDenominatorsPart2(n_occ_, n_vir_, eps_occ, eps_vir, eps_ijk, datatype_tapp);

    delete[] uuid_;
    return 0;
  }

  int distributed_create_tensor(TAPP_tensor_info info, void* init_val){
    initialize();

    /*if(info == nullptr) {
      std::cout << "Error: nullptr exception in distributed_create_tensor in TAPP_tensor_info. " <<  std::endl;
      exit(445);
    }*/
    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);

    TAPP_datatype datatype_tapp = TAPP_get_datatype(info);
    int nmodes = TAPP_get_nmodes(info);

    int64_t* extents = new int64_t[nmodes]; 
    if(nmodes > 0) TAPP_get_extents(info, extents);

    std::complex<double> init_val_ = 0.0;
    if(datatype_tapp == TAPP_F64){
      init_val_ = std::complex<double>(*static_cast<double*>(init_val));
    }
    else if(datatype_tapp == TAPP_C64){
      init_val_ = *static_cast<std::complex<double>*>(init_val);
    }
    else {
      std::cout << "Error: distributed_create_tensor with the requested datatype not implemented with Cyclops CTF. " <<  std::endl;
      exit(489);
    }
     
    std::string name = ""; 

    createTensor(uuid, nmodes, extents, datatype_tapp, name, init_val_);
    
    delete[] uuid_;
    delete[] extents;
    return 0;
  }

  int distributed_destruct_tensor(TAPP_tensor_info info){
    initialize();
    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);
    
    destructTensor(uuid);
    delete[] uuid_;
    return 0;
  }

  int distributed_broadcast_tensor_body_data(TAPP_tensor_info info, void* data){
    TAPP_datatype datatype_tapp = TAPP_get_datatype(info);

    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);

    distributeArrayDataPart1(uuid);
    distributeArrayDataPart2(data, 0, nullptr, nullptr, datatype_tapp);

    delete[] uuid_;
    return 0;
  }

  int distributed_gather_tensor_body_data(TAPP_tensor_info info, void* data){
    TAPP_datatype datatype_tapp = TAPP_get_datatype(info);

    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);

    gatherDistributedArrayDataPart1(uuid);
    gatherDistributedArrayDataPart2(data, 0, nullptr, nullptr, datatype_tapp);

    delete[] uuid_;
    return 0;
  }

  int sctest(TAPP_tensor_info info){
    TAPP_datatype datatype_tapp = TAPP_get_datatype(info);

    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);



    //test
    int nmodes = TAPP_get_nmodes(info);
    if(nmodes == 2 || nmodes == 4 || nmodes == 6 || nmodes == 8){
      int64_t* extents = new int64_t[nmodes]; 
      TAPP_get_extents(info, extents);
      if ((nmodes == 2) || (nmodes == 4 && extents[0] == extents[1] && extents[2] == extents[3])){
        std::cout << " GOING in  --------------------- " <<  std::endl;
			  std::complex<double> * occ = new std::complex<double>[extents[nmodes-1]]; 
			  std::complex<double> * vir = new std::complex<double>[extents[0]]; 
				std::random_device rd;
				std::mt19937 gen(rd());
				std::uniform_real_distribution<> dis(0.001, 0.5);
				for(int i=0; i<extents[nmodes-1]; i++ ) occ[i] = dis(gen);
				for(int i=0; i<extents[0]; i++ ) vir[i] = dis(gen);
			  std::complex<double> eps_ijk = 0.0;
			  ierr = distributed_scale_with_denominators(info, extents[nmodes-1], extents[0], occ, vir,  &eps_ijk);
        std::cout << " done  --------------------- " <<  std::endl;
      }
    }

    delete[] uuid_;
    return 0;

  }

  int distributed_tensor_set_name(TAPP_tensor_info info, const int64_t* name, const int name_len){
    TAPP_datatype datatype_tapp = TAPP_get_datatype(info);

    int uuid_len = TAPP_get_uuid_len(info); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr = TAPP_get_uuid(info, uuid_, uuid_len);
    std::string uuid(uuid_);

    int64_t* buffi = new int64_t[name_len];

    std::memcpy(buffi, name, sizeof(buffi));

    char* buffc = new char[name_len + 1];
    for (int i = 0; i < name_len; i++) {
    	buffc[i] = static_cast<char>(buffi[i]); 
    }
    buffc[name_len] = '\0'; // null terminator

    std::string name_(buffc);

    tensorSetName(uuid, name_);

    delete[] uuid_;
    delete[] buffi;
    delete[] buffc;
    return 0;
  }

  int initializeWork(){
    return initialize_();
  }

  int finalizeWork(){
    return finalize();
  }

  int ctf_bind_execute_product(TAPP_tensor_info info_A, void* A, int op_A, int64_t* idx_A,
  			TAPP_tensor_info info_B, void* B, int op_B, int64_t* idx_B,
  			TAPP_tensor_info info_C, void* C, int op_C, int64_t* idx_C,
  			TAPP_tensor_info info_D, void* D, int op_D, int64_t* idx_D,
  			void* alpha, void* beta){
    initialize();


    TAPP_datatype datatype_tapp = TAPP_get_datatype(info_A);

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

    int uuid_len = TAPP_get_uuid_len(info_A); 
    char* uuid_ = new char[uuid_len + 1];
    int ierr;

    //name = 'A';
    ierr = TAPP_get_uuid(info_A, uuid_, uuid_len);
    std::string uuid_A(uuid_);
   
    //name = 'B';
    ierr = TAPP_get_uuid(info_B, uuid_, uuid_len);
    std::string uuid_B(uuid_);
   
    //name = 'C';
    ierr = TAPP_get_uuid(info_C, uuid_, uuid_len);
    std::string uuid_C(uuid_);
   
    //name = 'D';
    ierr = TAPP_get_uuid(info_D, uuid_, uuid_len);
    std::string uuid_D(uuid_);
   
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


    int nmodes_A = TAPP_get_nmodes(info_A);
    int nmodes_B = TAPP_get_nmodes(info_B);
    int nmodes_C = TAPP_get_nmodes(info_C);
    int nmodes_D = TAPP_get_nmodes(info_D);
   
    executeProduct(uuid_A, nmodes_A, idx_A, op_A, uuid_B, nmodes_B, idx_B, op_B, 
                   uuid_C, nmodes_C, idx_C, op_C, uuid_D, nmodes_D, idx_D, op_D, alpha_, beta_);
   
   
    /* //comm test
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
    */
    // std::cout << " op_D " << op_D << " op_A " << op_A << " op_B " << op_B << " op_C " << op_C << std::endl;
   
    //destructTensor(uuid_A);
    //destructTensor(uuid_B);
    //destructTensor(uuid_C);
    //if (uuid_C != uuid_D) destructTensor(uuid_D);
    // finalize();
    // exit(0);

    delete[] uuid_;
    return 0;
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


