
#include "mpi_utils.h"
#include <ctf.hpp>
using namespace CTF;



// compute a single Jacobi iteration to get new x, elementwise: x_i <== d_i*(b_i-sum_j R_ij*x_j)
// solves Ax=b where R_ij=A_ij for i!=j, while R_ii=0, and d_i=1/A_ii
void jacobi_iter(Matrix<> & R, Vector<> & b, Vector<> & d, Vector<> &x){
  x["i"] = -R["ij"]*x["j"];
  x["i"] += b["i"];
  x["i"] *= d["i"];
}

int jacobi(int     n,
           World & dw){

  if (dw.rank == 0){
    printf("running jacobi method on random %d-by-%d sparse matrix\n",n,n);
  }

  std::cout << "sep:" << std::endl;
  Matrix<> spA(n, n, SP, dw, "spA");
  Matrix<> dnA(n, n, dw, "dnA");
  Vector<> b(n, dw);
  Vector<> c1(n, dw);
  Vector<> c2(n, dw);
  Vector<> res(n, dw);

  srand48(dw.rank);
  b.fill_random(0.0,1.0);
  c1.fill_random(0.0,1.0);
  c2["i"] = c1["i"];

  //make diagonally dominant matrix
  dnA.fill_random(0.0,1.0);
  spA["ij"] += dnA["ij"];
  //sparsify
  spA.sparsify(.5);
  spA["ii"] += 2.*n;
  dnA["ij"] = spA["ij"];

    int64_t  sz, * indices;
    double * values;
  dnA.get_local_data(&sz, &indices, &values);
  std::cout << "sep1 sz:" << sz << std::endl;

  Vector<> d(n, dw);
  d["i"] = spA["ii"];
  Transform<> inv([](double & d){ d=1./d; });
  inv(d["i"]);
  
  Matrix<> spR(n, n, SP, dw, "spR");
  Matrix<> dnR(n, n, dw, "dnR");
  spR["ij"] = spA["ij"];
  dnR["ij"] = dnA["ij"];
  spR["ii"] = 0;
  dnR["ii"] = 0;

/*  spR.print(); 
  dnR.print(); */
 
  //do up to 100 iterations
  double res_norm;
  int iter;
  for (iter=0; iter<100; iter++){
    jacobi_iter(dnR, b, d, c1);

    res["i"]  = b["i"];
    res["i"] -= dnA["ij"]*c1["j"];

    res_norm = res.norm2();
    if (res_norm < 1.E-4) break;
  }
  if (dw.rank == 0)
    printf("Completed %d iterations of Jacobi with dense matrix, residual F-norm is %E\n", iter, res_norm);

  for (iter=0; iter<100; iter++){
    jacobi_iter(spR, b, d, c2);

    res["i"]  = b["i"];
    res["i"] -= spA["ij"]*c2["j"];

    res_norm = res.norm2();
    if (res_norm < 1.E-4) break;
  }
  if (dw.rank == 0)
    printf("Completed %d iterations of Jacobi with sparse matrix, residual F-norm is %E\n", iter, res_norm);

  c2["i"] -= c1["i"];

  bool pass = c2.norm2() <= 1.E-6;

  if (dw.rank == 0){
    if (pass) 
      printf("{ Jacobi x[\"i\"] = (1./A[\"ii\"])*(b[\"j\"] - (A[\"ij\"]-A[\"ii\"])*x[\"j\"]) with sparse A } passed \n");
    else
      printf("{ Jacobi x[\"i\"] = (1./A[\"ii\"])*(b[\"j\"] - (A[\"ij\"]-A[\"ii\"])*x[\"j\"]) with sparse A } failed \n");
  }
  
  waitWorkersFinished();
  return pass;
} 



int* int64ToIntArray(const int64_t* input, size_t size) {
    if (input == nullptr || size == 0) {
        return nullptr; // Handle invalid input
    }

    int* output = new int[size]; 

    for (size_t i = 0; i < size; ++i) {
        output[i] = static_cast<int>(input[i]);
    }

    return output;
}

int initializeMPIandWorkerGroup(MPI_Group& new_group, MPI_Comm& new_comm, int& new_rank){
  int rank0, np;
  int initialized, finalized;

  std::cout << "calling mpi init sep:" << std::endl;
  MPI_Initialized(&initialized);

  std::cout << "pt2 sep:" << std::endl;
  if (!initialized) {
  	MPI_Init(nullptr, nullptr);
  }
  std::cout << "pt3 sep:" << std::endl;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank0);
  MPI_Comm_size(MPI_COMM_WORLD, &np);
  std::cout << "pt4 sep:" << std::endl;


  std::cout << "np1_before:" << np << std::endl;
  std::cout << "rank1_before:" << rank0 << std::endl;

  std::cout << "got to dw -2"  << std::endl;
  MPI_Group world_group;
  std::cout << "got to dw 0"  << std::endl;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int ranks_to_exclude[] = {0}; // exclude ranks 0 and 2
  int num_ranks_to_exclude = 1;

  std::cout << "got to dw 1"  << std::endl;
  MPI_Group_excl(world_group, num_ranks_to_exclude, ranks_to_exclude, &new_group);

  std::cout << "got to dw 2"  << std::endl;
  MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

  std::cout << "got to dw 3"  << std::endl;
  MPI_Comm_rank(new_comm, &new_rank);
  MPI_Comm_size(new_comm, &np);
  std::cout << "np1_after:" << np << std::endl;
  std::cout << "rank1_after:" << new_rank << std::endl;

  return 0;
}


int createTensor_(World& dw, std::map<std::string, Tensor<>>& tensorR, std::map<std::string, Tensor<std::complex<double>>>& tensorC){
  // int64_t nmodes = mpiBroadcastInt();
  std::string uuid;
  int nmodes = -1;
  int64_t* extents = nullptr;
  int datatype_tapp;
  std::string name;
  std::complex<double> init_val;

  createTensor(uuid, nmodes, extents, datatype_tapp, name, init_val);

  // all data transmitted
  if(dw.rank == 0){
    std::cout << "nmodes: " << nmodes << std::endl;
    for (int i = 0; i < nmodes; ++i) std::cout << "val " << extents[i] << std::endl;
    std::cout << "uuid: " << uuid << std::endl;
  }
  int* extents_ = int64ToIntArray(extents, nmodes);
  

  //int shape[] = {NS,NS,NS,NS};
  int* shape = new int[nmodes];
  for (int i = 0; i < nmodes; ++i){
    shape[i] = NS;
  }
  //* Creates a distributed tensor initialized to zero
  switch (datatype_tapp) { // tapp_datatype
    case 1://TAPP_R64:
      {
        Tensor<> tens(nmodes, extents_, shape, dw);
        if(!name.empty()) tens.set_name(name.c_str());
        if(init_val != 0.0) tens = init_val.real();
        tensorR[uuid] = tens;

        //* Writes noise to local data based on global index
        int64_t  numel, * indices;
        double * values;
        tensorR[uuid].get_local_data(&numel, &indices, &values);
        for (int i=0; i<numel; i++ ) values[i] = drand48()-.5;
      }
      break;
    case 3://TAPP_C64:
      {
        Tensor<std::complex<double>> tens(nmodes, extents_, shape, dw);
        if(!name.empty()) tens.set_name(name.c_str());
        if(init_val != 0.0) tens = init_val;
        tensorC[uuid] = tens;

        //* Writes noise to local data based on global index
        int64_t  numel, * indices;
        std::complex<double> * values;
        tensorC[uuid].get_local_data(&numel, &indices, &values);
        for (int i=0; i<numel; i++ ) values[i] = drand48()-.5;
      }
      break;
  }

  delete[] extents;
  delete[] extents_;
  delete[] shape;

  waitWorkersFinished();
  return 0;
}


int distributeArrayData(World& dw, std::map<std::string, Tensor<>>& tensorR, std::map<std::string, Tensor<std::complex<double>>>& tensorC){
  std::string uuid;
  int datatype_tapp;

  distributeArrayDataPart1(uuid);

  if(tensorR.find(uuid) != tensorR.end()) datatype_tapp = 1;
  else if(tensorC.find(uuid) != tensorC.end()) datatype_tapp = 3; 
  else {
    std::cout << "ERROR: Tensor not found. uuid: " << uuid<< std::endl;
    exit(404);
  }

  int np;
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  int64_t  numel, * indices;
  double * valuesR;
  std::complex<double> * valuesC;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(datatype_tapp == 1){
      tensorR[uuid].get_local_data(&numel, &indices, &valuesR);
  }
  else if(datatype_tapp == 3){
      tensorC[uuid].get_local_data(&numel, &indices, &valuesC);
  } 
  waitWorkersFinished();
  
  if(datatype_tapp == 1){
      double* full_dat;
      distributeArrayDataPart2(full_dat, numel, indices, valuesR, datatype_tapp);
  }
  else if(datatype_tapp == 3){
      std::complex<double>* full_dat;
      distributeArrayDataPart2(full_dat, numel, indices, valuesC, datatype_tapp);
      for(int i=0;i<numel;i++) std::cout << "Aw " << valuesC[i] << std::endl;
  } 

  waitWorkersFinished();
  return 0;
}

int main(int argc, char ** argv){

  MPI_Group new_group;
  MPI_Comm new_comm;
  int new_rank;
  initializeMPIandWorkerGroup(new_group, new_comm, new_rank);

  { 
    printf("got to dw \n");
    int  in_num = 3;
    char ** input_str = new char*[in_num];;
    input_str[0] = "test++";
    input_str[1] = "-n";
    input_str[2] = "1000";

    World dw0(in_num, input_str);
    World dw(new_comm, in_num, input_str);

    std::map<std::string, Tensor<>> tensorR;
    std::map<std::string, Tensor<std::complex<double>>> tensorC;
    
    bool run = true; 
    while(run){
      std::string message;
      mpiBroadcastString(message);
      
      int ierr = -1;
      if(message == "doJacobi"){
        ierr = (int)(!jacobi(1000, dw));
      }
      else if(message == "createTensor"){
        ierr = createTensor_(dw, tensorR, tensorC);
      }
      else if(message == "distributeArrayData"){
        ierr = distributeArrayData(dw, tensorR, tensorC);
      }
      else if(message == "stopWorker"){
        run = false;
        waitWorkersFinished();
        ierr = 0;
      }
      else{
        std::cout << "ERROR: worker recieved uknown message: " << message << std::endl;
        exit(500);
      } 
      assert((ierr == 0));

    }
  }
  MPI_Group_free(&new_group);
  MPI_Comm_free(&new_comm);
  int finalized;
  MPI_Finalized(&finalized);
  if(!finalized){
    MPI_Finalize();
  }
  return 0;
}

