
#include "mpi_utils.h"
#include <ctf.hpp>
#include <memory>
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

    for (size_t i = 0; i < size; i++) {
        output[i] = static_cast<int>(input[i]);
    }

    return output;
}

char* int64_tArrayToCharArray(const int64_t* input, size_t size) {
    if (size == 0) {
        char* emptyStr = new char[1]; // Allocate space for null terminator
        emptyStr[0] = '\0';
        return emptyStr;
    }

    char* charArray = new char[size + 1]; // +1 for null terminator
    for (size_t i = 0; i < size; i++) {
        charArray[i] = static_cast<char>(input[i]);
    }
    charArray[size] = '\0'; // Null-terminate the string
    return charArray;
}

void printCTensor(Tensor<std::complex<double>>& tens, World& dw){
  int64_t  numel, * indices;
  std::complex<double> * values;
  tens.get_local_data(&numel, &indices, &values);
  if(dw.rank == 0) std::cout << "Printing tensor " << tens.get_name() << ":"<< std::endl;
  for (int i=0; i<numel; i++ ) std::cout << values[i] << std::endl;
}

int initializeMPIandWorkerGroup(MPI_Group& new_group, MPI_Comm& new_comm, int& new_rank){
  int rank0, np;
  int initialized, finalized;

  std::cout << "worker mpi init" << std::endl;
  MPI_Initialized(&initialized);

  if (!initialized) {
  	MPI_Init(nullptr, nullptr);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank0);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  MPI_Group world_group;
  MPI_Comm_group(MPI_COMM_WORLD, &world_group);
  int ranks_to_exclude[] = {0}; // exclude ranks 0 and 2
  int num_ranks_to_exclude = 1;

  MPI_Group_excl(world_group, num_ranks_to_exclude, ranks_to_exclude, &new_group);

  MPI_Comm_create(MPI_COMM_WORLD, new_group, &new_comm);

  MPI_Comm_rank(new_comm, &new_rank);
  MPI_Comm_size(new_comm, &np);

  return 0;
}


int createTensor_(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC){
  // int64_t nmodes = mpiBroadcastInt();
  std::string uuid;
  int nmodes = -1;
  int64_t* extents = nullptr;
  int datatype_tapp;
  std::string name;
  std::complex<double> init_val;

  createTensor(uuid, nmodes, extents, datatype_tapp, name, init_val);

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
        std::unique_ptr<Tensor<>> tens = std::make_unique<Tensor<>>(nmodes, extents_, shape, dw);
        if(!name.empty()) tens->set_name(name.c_str());
        if(init_val != 0.0) {
          *tens = init_val.real();
        }
        tensorR[uuid] = std::move(tens);
      }
      break;
    case 3://TAPP_C64:
      {
        // Ring< std::complex<double> > dring;
        std::unique_ptr<Tensor<std::complex<double>>> tens = std::make_unique<Tensor<std::complex<double>>>(nmodes, extents_, shape, dw); //, dring
        if(!name.empty()) tens->set_name(name.c_str());
        if(std::abs(init_val) != 0.0) {
          *tens = init_val;
        }
        tensorC[uuid] = std::move(tens);

        /*std::unique_ptr<Tensor<>> tensr = std::make_unique<Tensor<>>(nmodes, extents_, shape, dw);
        if(init_val != 0.0) {
          *tensr = init_val.real();
        }
        tensorR[uuid+std::string("_r")] = std::move(tensr);

        std::unique_ptr<Tensor<>> tensi = std::make_unique<Tensor<>>(nmodes, extents_, shape, dw);
        if(init_val != 0.0) {
          *tensi = init_val.imag();
        }
        tensorR[uuid+std::string("_i")] = std::move(tensi);*/
        
      }
      break;
  }

  delete[] extents;
  delete[] extents_;
  delete[] shape;

  waitWorkersFinished();
  return 0;
}

int initializeTensor_(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC){
  int datatype_tapp;
  std::string uuid;
  std::complex<double> init_val;

  initializeTensor(uuid, init_val);
  
  if(tensorR.find(uuid) != tensorR.end()) datatype_tapp = 1;
  else if(tensorC.find(uuid) != tensorC.end()) datatype_tapp = 3; 
  else {
    std::cout << "ERROR: Tensor not found. uuid: " << uuid<< std::endl;
    exit(404);
  }

  switch (datatype_tapp) { // tapp_datatype
    case 1://TAPP_R64:
      *tensorR[uuid] = init_val.real();
      break;
    case 3://TAPP_C64:
      *tensorC[uuid] = init_val;
      break;
  }

  waitWorkersFinished();
  return 0;
}

int distributeArrayData(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC, bool gather){
  std::string uuid;
  int datatype_tapp;
  
  if(gather){
    gatherDistributedArrayDataPart1(uuid);
  }
  else{
    distributeArrayDataPart1(uuid);
  }

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
      tensorR[uuid]->get_local_data(&numel, &indices, &valuesR);
  }
  else if(datatype_tapp == 3){
      tensorC[uuid]->get_local_data(&numel, &indices, &valuesC);
  } 
  waitWorkersFinished();
  
  if(datatype_tapp == 1){
      double* full_dat;

      if(gather) {
        gatherDistributedArrayDataPart2(full_dat, numel, indices, valuesR, datatype_tapp);
      }
      else {
        distributeArrayDataPart2(full_dat, numel, indices, valuesR, datatype_tapp);
        tensorR[uuid]->write(numel, indices, valuesR);
      }
  }
  else if(datatype_tapp == 3){
      std::complex<double>* full_dat;

      if(gather) {
        gatherDistributedArrayDataPart2(full_dat, numel, indices, valuesC, datatype_tapp);
      }
      else {
        distributeArrayDataPart2(full_dat, numel, indices, valuesC, datatype_tapp);
        tensorC[uuid]->write(numel, indices, valuesC);
        
        /*int64_t  numelr, * indicesr;
        double * valuesr;
        tensorR[uuid+std::string("_r")]->get_local_data(&numelr, &indicesr, &valuesr);
        if(numel != numelr) {
          std::cout << "ERROR: numel != numelr " << std::endl;
          exit(404);
        }
        for(int64_t i = 0; i<numel; i++) valuesr[i] = valuesC[i].real(); 
        tensorR[uuid+std::string("_r")]->write(numelr, indicesr, valuesr);

        int64_t  numeli, * indicesi;
        double * valuesi;
        tensorR[uuid+std::string("_i")]->get_local_data(&numeli, &indicesi, &valuesi);
        if(numel != numeli) {
          std::cout << "ERROR: numel != numeli " << std::endl;
          exit(404);
        }
        for(int64_t i = 0; i<numel; i++) valuesi[i] = valuesC[i].imag(); 
        tensorR[uuid+std::string("_i")]->write(numeli, indicesi, valuesi); */
      }
      // for(int i=0;i<numel;i++) std::cout << "Aw " << valuesC[i] << std::endl;
      if(dw.rank == 0){
        if(gather) std::cout << "gather " ;
        else std::cout << "distribute " ;
      }
      printCTensor(*(tensorC[uuid]), dw);
  } 

  waitWorkersFinished();
  return 0;
}

int copyTensor_(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC){
  int datatype_dest, datatype_src;
  std::string dest, src;

  copyTensor(dest, src);

  if(tensorR.find(dest) != tensorR.end()) datatype_dest = 1;
  else if(tensorC.find(dest) != tensorC.end()) datatype_dest = 3; 
  else {
    std::cout << "ERROR: Tensor not found. uuid: " << dest << std::endl;
    exit(404);
  }
  if(tensorR.find(src) != tensorR.end()) datatype_src = 1;
  else if(tensorC.find(src) != tensorC.end()) datatype_src = 3; 
  else {
    std::cout << "ERROR: Tensor not found. uuid: " << src << std::endl;
    exit(404);
  }
  if (datatype_dest != datatype_src){
    std::cout << "ERROR: Copied tensors cannot have different datatypes. " << std::endl;
    exit(403);
  }

  switch (datatype_src) { // tapp_datatype
    case 1://TAPP_R64:
      *tensorR[dest] = *tensorR[src];
      break;
    case 3://TAPP_C64:
      *tensorC[dest] = *tensorC[src];
      break;
  }

  waitWorkersFinished();
  return 0;
}

int tensorSetName_(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC){
  int datatype_tapp;
  std::string uuid;
  std::string name;

  tensorSetName(uuid, name);

  if(tensorR.find(uuid) != tensorR.end()) datatype_tapp = 1;
  else if(tensorC.find(uuid) != tensorC.end()) datatype_tapp = 3; 
  else {
    std::cout << "ERROR: Tensor not found. uuid: " << uuid<< std::endl;
    exit(404);
  }

  switch (datatype_tapp) { // tapp_datatype
    case 1://TAPP_R64:
      tensorR[uuid]->set_name(name.c_str());
      break;
    case 3://TAPP_C64:
      tensorC[uuid]->set_name(name.c_str());
      break;
  }

  waitWorkersFinished();
  return 0;
}

int destructTensor_(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC){
  // int64_t nmodes = mpiBroadcastInt();
  std::string uuid;
  int datatype;

  destructTensor(uuid);

  if(tensorR.find(uuid) != tensorR.end()) datatype = 1;
  else if(tensorC.find(uuid) != tensorC.end()) datatype = 3; 
  else { std::cout << "ERROR: Attempting to destruct nonexisting tensor. uuid: " << uuid << std::endl; exit(404); }
  
  if(datatype == 1){
    tensorR[uuid]->free_self();
    tensorR.erase(uuid);
  }
  else if(datatype == 3){
    tensorC[uuid]->free_self();
    tensorC.erase(uuid);
  } 

  waitWorkersFinished();
  return 0;
}

int scaleWithDenominators_(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC){
  std::string uuid;
  int datatype_tapp, n_occ, n_vir;

  scaleWithDenominatorsPart1(uuid, n_occ, n_vir);

  if(tensorR.find(uuid) != tensorR.end()) datatype_tapp = 1;
  else if(tensorC.find(uuid) != tensorC.end()) datatype_tapp = 3; 
  else { std::cout << "ERROR: Attempting to destruct nonexisting tensor. uuid: " << uuid << std::endl; exit(404); }

  int64_t* extentso = new int64_t[1];
  extentso[0] = n_occ;
  int64_t* extentsv = new int64_t[1];
  extentso[0] = n_vir;
  int* shape = new int[1];
  shape[0] = NS;
  //* Creates a distributed tensor initialized to zero
  switch (datatype_tapp) { // tapp_datatype
    case 1://TAPP_R64:
      {
        Tensor<> eps_occ_t = Tensor<>(1, extentso, shape, dw);
        Tensor<> eps_vir_t = Tensor<>(1, extentsv, shape, dw);
        eps_occ_t.set_name("eps_occ_t");
        eps_vir_t.set_name("eps_vir_t");
 	      double* eps_occ = new double[n_occ];
 	      double* eps_vir = new double[n_vir];
 	      double* eps_ijk = new double[1];
        waitWorkersFinished();
        scaleWithDenominatorsPart2(n_occ, n_vir, eps_occ, eps_vir, eps_ijk, datatype_tapp);
    
        int64_t  numelo, * indiceso;
        double * valueso;
        int64_t  numelv, * indicesv;
        double * valuesv;
        eps_occ_t.get_local_data(&numelo, &indiceso, &valueso);
        eps_vir_t.get_local_data(&numelv, &indicesv, &valuesv);
        for(int i = 0; i<numelo, i++) valueso[i] = eps_occ[indiceso[i]];
        for(int i = 0; i<numelv, i++) valuesv[i] = eps_vir[indicesv[i]];
        eps_occ_t.write(numelo, indiceso, valueso);
        eps_vir_t.write(numelv, indicesv, valuesv);
        
  	    Tensor<> denom(*tensorR[uuid]);
        denom = eps_ijk[0];

        CTF::Scalar<> one_t(dw);
        one_t = 1.0;

        double divide(double a, double b){
          return a/b;
        }
  	    Function<> fctr(&divide);

				std::string idx_ = "";
				int order = denom.order;
				if(order == 2) {
					idx_ = "ai";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"];
  	      denom[idx] -= eps_vir_t["a"];
				}
				else if(order == 4) {
					idx_ = "abij";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"];
				}
				else if(order == 6) {
					idx_ = "abcijk";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"]; denom[idx] += eps_occ_t["k"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"]; denom[idx] -= eps_vir_t["c"];
				}
				else if(order == 8) {
					idx_ = "abcdijkl";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"];
  	      denom[idx] += eps_occ_t["k"]; denom[idx] += eps_occ_t["l"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"];
  	      denom[idx] -= eps_vir_t["c"]; denom[idx] -= eps_vir_t["d"];
				}
				else if(order == 10) {
					idx_ ="abcdeijklm";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"]; denom[idx] += eps_occ_t["k"];
  	      denom[idx] += eps_occ_t["l"]; denom[idx] += eps_occ_t["m"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"];i denom[idx] -= eps_vir_t["c"];
  	      denom[idx] -= eps_vir_t["d"]; denom[idx] -= eps_vir_t["e"];
				}
  	      tensorR[uuid]->contract(1.0, one_t, "", denom, idx_.c_str(), 0.0, idx_.c_str(), fctr);

        denom->free_self();
        one_t->free_self();
        eps_occ_t->free_self();
        eps_vir_t->free_self();
        delete[] eps_occ;
        delete[] eps_vir;
        delete[] eps_ijk;
      }
    break;
    case 3://TAPP_C64:
      {
       
        Tensor<std::complex<double>> eps_occ_t = Tensor<std::complex<double>>(1, extentso, shape, dw);
        Tensor<std::complex<double>> eps_vir_t = Tensor<std::complex<double>>(1, extentsv, shape, dw);
        eps_occ_t.set_name("eps_occ_t");
        eps_vir_t.set_name("eps_vir_t");
 	      std::complex<double>* eps_occ = new std::complex<double>[n_occ];
 	      std::complex<double>* eps_vir = new std::complex<double>[n_vir];
 	      std::complex<double>* eps_ijk = new std::complex<double>[1];
        waitWorkersFinished();
        scaleWithDenominatorsPart2(n_occ, n_vir, eps_occ, eps_vir, eps_ijk, datatype_tapp);
    
        int64_t  numelo, * indiceso;
        std::complex<double> * valueso;
        int64_t  numelv, * indicesv;
        std::complex<double> * valuesv;
        eps_occ_t.get_local_data(&numelo, &indiceso, &valueso);
        eps_vir_t.get_local_data(&numelv, &indicesv, &valuesv);
        for(int i = 0; i<numelo, i++) valueso[i] = eps_occ[indiceso[i]];
        for(int i = 0; i<numelv, i++) valuesv[i] = eps_vir[indicesv[i]];
        eps_occ_t.write(numelo, indiceso, valueso);
        eps_vir_t.write(numelv, indicesv, valuesv);
        
  	    Tensor<std::complex<double>> denom(*tensorC[uuid]);
        denom = eps_ijk[0];

        // CTF::Scalar<std::complex<double>> eps_ijk_t(dw);
        // eps_ijk_t = eps_ijk[0];
        CTF::Scalar<std::complex<double>> one_t(dw);
        one_t = std::complex<double>(1.0, 0.0);

        std::complex<double> divide(std::complex<double> a, std::complex<double> b){
          return a/b;
        }
  	    Function<std::complex<double>> fctr(&divide);

				std::string idx_ = "";
				int order = denom.order;
				if(order == 2) {
					idx_ = "ai";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"];
  	      denom[idx] -= eps_vir_t["a"];
				}
				else if(order == 4) {
					idx_ = "abij";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"];
				}
				else if(order == 6) {
					idx_ = "abcijk";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"]; denom[idx] += eps_occ_t["k"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"]; denom[idx] -= eps_vir_t["c"];
				}
				else if(order == 8) {
					idx_ = "abcdijkl";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"];
  	      denom[idx] += eps_occ_t["k"]; denom[idx] += eps_occ_t["l"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"];
  	      denom[idx] -= eps_vir_t["c"]; denom[idx] -= eps_vir_t["d"];
				}
				else if(order == 10) {
					idx_ ="abcdeijklm";
                    const char * idx = idx_.c_str();
  	      denom[idx] += eps_occ_t["i"]; denom[idx] += eps_occ_t["j"]; denom[idx] += eps_occ_t["k"];
  	      denom[idx] += eps_occ_t["l"]; denom[idx] += eps_occ_t["m"];
  	      denom[idx] -= eps_vir_t["a"]; denom[idx] -= eps_vir_t["b"];i denom[idx] -= eps_vir_t["c"];
  	      denom[idx] -= eps_vir_t["d"]; denom[idx] -= eps_vir_t["e"];
				}
  	      tensorC[uuid]->contract(1.0, one_t, "", denom, idx_.c_str(), 0.0, idx_.c_str(), fctr);

        denom->free_self();
        one_t->free_self();
        eps_occ_t->free_self();
        eps_vir_t->free_self();
        delete[] eps_occ;
        delete[] eps_vir;
        delete[] eps_ijk;
      }
    break;
  } 

  delete[] shape;
  delete[] extentso;
  delete[] extentsv;

  waitWorkersFinished();
  return 0;
}

int executeProduct_(World& dw, std::map<std::string, std::unique_ptr<Tensor<>>>& tensorR, std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>>& tensorC){
  std::string uuid_A, uuid_B, uuid_C, uuid_D;
  int nmode_A, nmode_B, nmode_C, nmode_D;
  int64_t * idx_A_ = nullptr, * idx_B_ = nullptr, * idx_C_ = nullptr, * idx_D_ = nullptr;
  std::complex<double> alpha, beta;

  executeProduct(uuid_A, nmode_A, idx_A_, uuid_B, nmode_B, idx_B_, uuid_C, nmode_C, idx_C_, uuid_D, nmode_D, idx_D_, alpha, beta);
  
  int datatype_A, datatype_B, datatype_C, datatype_D;
  if(tensorR.find(uuid_A) != tensorR.end()) datatype_A = 1;
  else if(tensorC.find(uuid_A) != tensorC.end()) datatype_A = 3; 
  else { std::cout << "ERROR: Tensor not found. uuid: " << uuid_A << std::endl; exit(404); }
  if(tensorR.find(uuid_B) != tensorR.end()) datatype_B = 1;
  else if(tensorC.find(uuid_B) != tensorC.end()) datatype_B = 3; 
  else { std::cout << "ERROR: Tensor not found. uuid: " << uuid_B << std::endl; exit(404); }
  if(tensorR.find(uuid_C) != tensorR.end()) datatype_C = 1;
  else if(tensorC.find(uuid_C) != tensorC.end()) datatype_C = 3; 
  else { std::cout << "ERROR: Tensor not found. uuid: " << uuid_C << std::endl; exit(404); }
  if(tensorR.find(uuid_D) != tensorR.end()) datatype_D = 1;
  else if(tensorC.find(uuid_D) != tensorC.end()) datatype_D = 3; 
  else { std::cout << "ERROR: Tensor not found. uuid: " << uuid_D << std::endl; exit(404); }
  
  if(!(datatype_A == datatype_B && datatype_A == datatype_C && datatype_A == datatype_D)){ 
    std::cout << "ERROR: executeProduct with tensor datatypes different is possible, but not implemented in the interface."  << std::endl; 
    exit(407); 
  }

  char* idx_A = int64_tArrayToCharArray(idx_A_, nmode_A);
  char* idx_B = int64_tArrayToCharArray(idx_B_, nmode_B);
  char* idx_C = int64_tArrayToCharArray(idx_C_, nmode_C);
  char* idx_D = int64_tArrayToCharArray(idx_D_, nmode_D);

  if(dw.rank == 0)
    std::cout << "contract: D[" << idx_D << "] = " << alpha << " A[" << idx_A << "]*B[" << idx_B << "] + " << beta << " C[" << idx_C << "]" << std::endl;
  
  double alpha_ = alpha.real();
  double beta_ = beta.real();
  double alphai = alpha.imag();
  double betai = beta.imag();
  if(datatype_D == 1){

    if(uuid_C != uuid_D){
      (*(tensorR[uuid_D]))[idx_D] = (*(tensorR[uuid_C]))[idx_C]; 
    }
    tensorR[uuid_D]->contract(alpha_,  *(tensorR[uuid_A]), idx_A, *(tensorR[uuid_B]), idx_B, beta_, idx_D);
  } 
  else if(datatype_D == 3){
    if(uuid_C != uuid_D){
      (*(tensorC[uuid_D]))[idx_D] = (*(tensorC[uuid_C]))[idx_C]; 
    }
    tensorC[uuid_D]->contract(alpha,  *(tensorC[uuid_A]), idx_A, *(tensorC[uuid_B]), idx_B, beta, idx_D);


  //} 
  //else if(datatype_D == 3){
    /* if(uuid_C != uuid_D){
      (*(tensorR[uuid_D+std::string("_r")]))[idx_D] = beta_ * (*(tensorR[uuid_C+std::string("_r")]))[idx_C] - betai * (*(tensorR[uuid_C+std::string("_i")]))[idx_C]; 
      (*(tensorR[uuid_D+std::string("_i")]))[idx_D] = beta_ * (*(tensorR[uuid_C+std::string("_i")]))[idx_C] + betai * (*(tensorR[uuid_C+std::string("_r")]))[idx_C]; 
    }
    
    tensorR[uuid_D+std::string("_r")]->contract(alpha_,  *(tensorR[uuid_A+std::string("_r")]), idx_A, *(tensorR[uuid_B+std::string("_r")]), idx_B, 1.0, idx_D);
    tensorR[uuid_D+std::string("_i")]->contract(alpha_,  *(tensorR[uuid_A+std::string("_r")]), idx_A, *(tensorR[uuid_B+std::string("_i")]), idx_B, 1.0, idx_D);
    tensorR[uuid_D+std::string("_i")]->contract(alpha_,  *(tensorR[uuid_A+std::string("_i")]), idx_A, *(tensorR[uuid_B+std::string("_r")]), idx_B, 1.0, idx_D);
    tensorR[uuid_D+std::string("_r")]->contract(-alpha_,  *(tensorR[uuid_A+std::string("_i")]), idx_A, *(tensorR[uuid_B+std::string("_i")]), idx_B, 1.0, idx_D);

    tensorR[uuid_D+std::string("_i")]->contract(alphai,  *(tensorR[uuid_A+std::string("_r")]), idx_A, *(tensorR[uuid_B+std::string("_r")]), idx_B, 1.0, idx_D);
    tensorR[uuid_D+std::string("_r")]->contract(-alphai,  *(tensorR[uuid_A+std::string("_r")]), idx_A, *(tensorR[uuid_B+std::string("_i")]), idx_B, 1.0, idx_D);
    tensorR[uuid_D+std::string("_r")]->contract(-alphai,  *(tensorR[uuid_A+std::string("_i")]), idx_A, *(tensorR[uuid_B+std::string("_r")]), idx_B, 1.0, idx_D);
    tensorR[uuid_D+std::string("_i")]->contract(-alphai,  *(tensorR[uuid_A+std::string("_i")]), idx_A, *(tensorR[uuid_B+std::string("_i")]), idx_B, 1.0, idx_D);

    int64_t  numel, * indices;
    std::complex<double> * values;
    tensorC[uuid_D]->get_local_data(&numel, &indices, &values);
    int64_t  numelr, * indicesr;
    double * valuesr;
    // *tensorR[uuid_D+std::string("_r")] = 1.0;
    tensorR[uuid_D+std::string("_r")]->get_local_data(&numelr, &indicesr, &valuesr);
    int64_t  numeli, * indicesi;
    double * valuesi;
    tensorR[uuid_D+std::string("_i")]->get_local_data(&numeli, &indicesi, &valuesi);
    // for(int64_t i = 0; i<numel; i++) values[i] = std::complex<double>(valuesr[i], valuesi[i]); 
    // tensorC[uuid_D]->write(numel, indices, values);
    for(int64_t i = 0; i<numel; i++) if (std::abs(values[i] - std::complex<double>(valuesr[i], valuesi[i])) > 0.000005){
      std::cout << "Difference: " << values[i] << " " << std::complex<double>(valuesr[i], valuesi[i])  << std::endl;
      exit(431);
    }
    */ 
  } 
  
  delete[] idx_A;
  delete[] idx_B;
  delete[] idx_C;
  delete[] idx_D;

  waitWorkersFinished();
  return 0;
}

int main(int argc, char ** argv){

  MPI_Group new_group;
  MPI_Comm new_comm;
  int new_rank;
  initializeMPIandWorkerGroup(new_group, new_comm, new_rank);

  { 
    // printf("got to dw \n");
  	int  in_num = 1;
  	char ** input_str = new char*[in_num];
    input_str[0] = new char[1] {'\0'};

    World dw0(in_num, input_str);
    World dw(new_comm, in_num, input_str); // communicator excluding rank 0 (master)

		delete[] input_str[0];
		delete[] input_str;

    std::map<std::string, std::unique_ptr<Tensor<>>> tensorR;
    std::map<std::string, std::unique_ptr<Tensor<std::complex<double>>>> tensorC;
    
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
      else if(message == "initializeTensor"){
        ierr = initializeTensor_(dw, tensorR, tensorC);
      }
      else if(message == "distributeArrayData"){
        ierr = distributeArrayData(dw, tensorR, tensorC, false);
      }
      else if(message == "executeProduct"){
        ierr = executeProduct_(dw, tensorR, tensorC);
      }
      else if(message == "gatherDistributedArrayData"){
        ierr = distributeArrayData(dw, tensorR, tensorC, true);
      }
      else if(message == "tensorSetName"){
        ierr = tensorSetName_(dw, tensorR, tensorC);
      }
      else if(message == "copyTensor"){
        ierr = copyTensor_(dw, tensorR, tensorC);
      }
      else if(message == "destructTensor"){
        ierr = destructTensor_(dw, tensorR, tensorC);
      }
      else if(message == "scaleWithDenominators"){
        ierr = scaleWithDenominators_(dw, tensorR, tensorC);
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

