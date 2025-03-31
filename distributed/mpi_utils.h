/*
 * Jan Brandejs
 * Toulouse III - Paul Sabatier University, France - April 2025
 */

#ifndef MPI_UTILS_H
#define MPI_UTILS_H
#ifdef __cplusplus

#include <iostream>
#include <random>
#include <tuple>
#include <string>
#include <complex>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include <ctf.hpp>
#pragma GCC diagnostic pop


void mpiBroadcastInt(int& num) {
  int rootRank = 0;
  MPI_Bcast(&num, 1, MPI_INT, rootRank, MPI_COMM_WORLD);
}

void mpiBroadcastString(std::string& str) {
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int strSize;
  std::vector<char> buffer;

  if (rank == rootRank) {
    strSize = str.size() + 1; // +1 for null terminator
  }

  MPI_Bcast(&strSize, 1, MPI_INT, rootRank, MPI_COMM_WORLD);

  buffer.resize(strSize);

  if (rank == rootRank) {
    std::copy(str.begin(), str.end() + 1, buffer.begin()); // Copy including null terminator
  }

  MPI_Bcast(buffer.data(), strSize, MPI_CHAR, rootRank, MPI_COMM_WORLD);

  if (rank != rootRank) {
    str = std::string(buffer.begin(), buffer.end() - 1); // Construct string without null terminator
  }
}

void mpiBroadcastC64(std::complex<double>& num) {
  int rootRank = 0;
  MPI_Bcast(&num, 1, MPI_DOUBLE_COMPLEX, rootRank, MPI_COMM_WORLD);
}

void mpiBroadcastInt64_tArray(int64_t*& data, int& numel) {
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Bcast(&numel, 1, MPI_INT, rootRank, MPI_COMM_WORLD);

  if (rank != rootRank) {
    if (numel > 0) {
      data = new int64_t[numel];
    }
    else data = nullptr;
  }
  
  if (numel > 0) {
    MPI_Bcast(data, numel, MPI_INT64_T, rootRank, MPI_COMM_WORLD);
  }
}

int waitWorkersFinished(){
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(rank == rootRank){
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    std::vector<int> finished_flags(np - 1, 0); // Track worker completion
    MPI_Status status;
 
    for(int worker=1; worker<np; worker++) { // Loop until all workers finish
      int finished;
      MPI_Recv(&finished, 1, MPI_INT, worker, 0, MPI_COMM_WORLD, &status);
      int worker_rank = status.MPI_SOURCE;
 
      if (finished == 137 && finished_flags[worker_rank - 1] == 0) {
        // std::cout << "Master: Worker " << worker_rank << " has finished." << std::endl;
        finished_flags[worker_rank - 1] = 1; // Mark worker as finished
      }
      else{
        std::cout << "Master: Error, worker " << worker_rank << " received msg: " << finished << " uknown or too many." << std::endl;
        exit(405);
      }
    }
  }
  else{
      int finished = 137; // Signal completion
      MPI_Send(&finished, 1, MPI_INT, rootRank, 0, MPI_COMM_WORLD);
      // std::cout << "Worker " << rank << ": Finished an instruction and sent signal to master." << std::endl;
  }
  return 0;
}


int createTensor(std::string& uuid, int& nmodes, int64_t*& extents, int& datatype_tapp, std::string& name, std::complex<double>& init_val){
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == rootRank){
    std::string message = "createTensor";
    mpiBroadcastString(message);
  }

  mpiBroadcastString(uuid);
  mpiBroadcastInt64_tArray(extents, nmodes);
  mpiBroadcastInt(datatype_tapp);
  
  int isName = 0;
  if(rank == rootRank && !name.empty()) isName = 1;
  mpiBroadcastInt(isName);
  if(isName == 1) mpiBroadcastString(name);
  
  mpiBroadcastC64(init_val);

  //worker allocates tensor
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

int distributeArrayDataPart1(std::string& uuid){ // int64_t& numel, int& datatype_tapp, void*& data
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == rootRank){
    std::string message = "distributeArrayData";
    mpiBroadcastString(message);
  }

  mpiBroadcastString(uuid);

  //each worker prepares local piece
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

int distributeArrayDataPart2(void* full_data, int64_t local_numel, int64_t* local_idx, void* local_data, int& data_type){
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status status;

  if(rank == rootRank){
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    std::vector<int> finished_flags(np - 1, 0); // Track worker completion

    for(int worker=1; worker<np; worker++) { // Loop until all workers finish
      MPI_Recv(&data_type, 1, MPI_INT, worker, 0, MPI_COMM_WORLD, &status);

      int64_t numel;
      MPI_Recv(&numel, 1, MPI_INT64_T, worker, 0, MPI_COMM_WORLD, &status);
      int worker_rank = status.MPI_SOURCE;
 
      if (finished_flags[worker_rank - 1] == 0) {
        // std::cout << "Master: Worker " << worker_rank << " has sent size:" << numel << std::endl;
        finished_flags[worker_rank - 1] = 1; // Mark worker as finished
      }
      else{
        std::cout << "Master: Error: too many messages from worker " << worker_rank << " msg: " << numel << std::endl;
        exit(404);
      }
      
      if(numel > 0){
        int64_t* local_idx_ = new int64_t[numel];
        MPI_Recv(local_idx_, numel, MPI_INT64_T, worker, 0, MPI_COMM_WORLD, &status);
        
        if(data_type == 1){ 
          double* local_data_ = new double[numel];
          double* full_data_ = static_cast<double*>(full_data);

          for(int64_t k=0;k<numel;k++){
            local_data_[k] = full_data_[local_idx_[k]]; 
          }
          MPI_Send(local_data_, numel, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD);
          delete[] local_data_;
        }
        else if(data_type == 3){ 
          std::complex<double>* local_data_ = new std::complex<double>[numel];
          std::complex<double>* full_data_ = static_cast<std::complex<double>*>(full_data);

          for(int64_t k=0;k<numel;k++){
            local_data_[k] = full_data_[local_idx_[k]]; 
          }
          MPI_Send(local_data_, numel, MPI_DOUBLE_COMPLEX, worker, 0, MPI_COMM_WORLD);
          delete[] local_data_;
        }
        else{ 
          std::cout << "Master: Error: uknown datatype received from worker " << worker_rank << " msg: " << data_type << std::endl;
          exit(406);
        }
        delete[] local_idx_;
        local_idx_ = nullptr;
      }
      
    }
    // std::cout << "Master: All workers have sent data." << std::endl;
  }
  else{//worker
    MPI_Send(&data_type, 1, MPI_INT, rootRank, 0, MPI_COMM_WORLD);

    MPI_Send(&local_numel, 1, MPI_INT64_T, rootRank, 0, MPI_COMM_WORLD);

    // std::cout << "Worker " << rank << ": sent local data size to master: " << local_numel << std::endl;

    if(local_numel > 0){
      MPI_Send(local_idx, local_numel, MPI_INT64_T, rootRank, 0, MPI_COMM_WORLD);

      if(data_type == 1){ 
        double* local_data_ = static_cast<double*>(local_data);
        MPI_Recv(local_data_, local_numel, MPI_DOUBLE, rootRank, 0, MPI_COMM_WORLD, &status);
      }
      else if(data_type == 3){ 
        std::complex<double>* local_data_ = static_cast<std::complex<double>*>(local_data);
        MPI_Recv(local_data_, local_numel, MPI_DOUBLE_COMPLEX, rootRank, 0, MPI_COMM_WORLD, &status);
      }
    }
  }
 
  //workers put the data to tensors
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

int gatherDistributedArrayDataPart1(std::string& uuid){ // int64_t& numel, int& datatype_tapp, void*& data
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == rootRank){
    std::string message = "gatherDistributedArrayData";
    mpiBroadcastString(message);
  }

  mpiBroadcastString(uuid);

  //each worker prepares local piece
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

int gatherDistributedArrayDataPart2(void* full_data, int64_t local_numel, int64_t* local_idx, void* local_data, int& data_type){
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status status;

  if(rank == rootRank){
    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    std::vector<int> finished_flags(np - 1, 0); // Track worker completion

    for(int worker=1; worker<np; worker++) { // Loop until all workers finish
      MPI_Recv(&data_type, 1, MPI_INT, worker, 0, MPI_COMM_WORLD, &status);

      int64_t numel;
      MPI_Recv(&numel, 1, MPI_INT64_T, worker, 0, MPI_COMM_WORLD, &status);
      int worker_rank = status.MPI_SOURCE;
 
      if (finished_flags[worker_rank - 1] == 0) {
        // std::cout << "Master: Worker " << worker_rank << " has sent size:" << numel << std::endl;
        finished_flags[worker_rank - 1] = 1; // Mark worker as finished
      }
      else{
        std::cout << "Master: Error: too many messages from worker " << worker_rank << " msg: " << numel << std::endl;
        exit(404);
      }
      
      if(numel > 0){
        int64_t* local_idx_ = new int64_t[numel];
        MPI_Recv(local_idx_, numel, MPI_INT64_T, worker, 0, MPI_COMM_WORLD, &status);
        
        if(data_type == 1){ 
          double* local_data_ = new double[numel];
          double* full_data_ = static_cast<double*>(full_data);

          MPI_Recv(local_data_, numel, MPI_DOUBLE, worker, 0, MPI_COMM_WORLD, &status);

          for(int64_t k=0;k<numel;k++){
            full_data_[local_idx_[k]] = local_data_[k]; 
          }
          delete[] local_data_;
        }
        else if(data_type == 3){ 
          std::complex<double>* local_data_ = new std::complex<double>[numel];
          std::complex<double>* full_data_ = static_cast<std::complex<double>*>(full_data);

          MPI_Recv(local_data_, numel, MPI_DOUBLE_COMPLEX, worker, 0, MPI_COMM_WORLD, &status);

          for(int64_t k=0;k<numel;k++){
            full_data_[local_idx_[k]] = local_data_[k]; 
          }
          delete[] local_data_;
        }
        else{ 
          std::cout << "Master: Error: uknown datatype received from worker " << worker_rank << " msg: " << data_type << std::endl;
          exit(406);
        }
        delete[] local_idx_;
        local_idx_ = nullptr;
      }
      
    }
    // std::cout << "Master: All workers have sent data." << std::endl;
  }
  else{//worker
    MPI_Send(&data_type, 1, MPI_INT, rootRank, 0, MPI_COMM_WORLD);

    MPI_Send(&local_numel, 1, MPI_INT64_T, rootRank, 0, MPI_COMM_WORLD);

    // std::cout << "Worker " << rank << ": sent local data size to master: " << local_numel << std::endl;

    if(local_numel > 0){
      MPI_Send(local_idx, local_numel, MPI_INT64_T, rootRank, 0, MPI_COMM_WORLD);

      if(data_type == 1){ 
        double* local_data_ = static_cast<double*>(local_data);
        MPI_Send(local_data_, local_numel, MPI_DOUBLE, rootRank, 0, MPI_COMM_WORLD);
      }
      else if(data_type == 3){ 
        std::complex<double>* local_data_ = static_cast<std::complex<double>*>(local_data);
        MPI_Send(local_data_, local_numel, MPI_DOUBLE_COMPLEX, rootRank, 0, MPI_COMM_WORLD);
      }
    }
  }
 
  //workers put the data to tensors
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

int tensorSetName(std::string& uuid, std::string& name){
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == rootRank){
    std::string message = "tensorSetName";
    mpiBroadcastString(message);
  }

  mpiBroadcastString(uuid);
  
  int isName = 0;
  if(rank == rootRank && !name.empty()) isName = 1;
  mpiBroadcastInt(isName);
  if(isName == 1) mpiBroadcastString(name);
  else if(rank != rootRank) name = "";
  
  //worker sets tensor name
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

int destructTensor(std::string& uuid){
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == rootRank){
    std::string message = "destructTensor";
    mpiBroadcastString(message);
  }

  mpiBroadcastString(uuid);

  //worker destructs tensor
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

int executeProduct(std::string& uuid_A, int& nmode_A, int64_t*& idx_A, std::string& uuid_B, int& nmode_B, int64_t*& idx_B, std::string& uuid_C, int& nmode_C, int64_t*& idx_C, std::string& uuid_D, int& nmode_D, int64_t*& idx_D, std::complex<double>& alpha, std::complex<double>& beta){
  int rootRank = 0;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if(rank == rootRank){
    std::string message = "executeProduct";
    mpiBroadcastString(message);
  }

  mpiBroadcastString(uuid_A);
  mpiBroadcastInt(nmode_A);
  mpiBroadcastInt64_tArray(idx_A, nmode_A);
  mpiBroadcastString(uuid_B);
  mpiBroadcastInt(nmode_B);
  mpiBroadcastInt64_tArray(idx_B, nmode_B);
  mpiBroadcastString(uuid_C);
  mpiBroadcastInt(nmode_C);
  mpiBroadcastInt64_tArray(idx_C, nmode_C);
  mpiBroadcastString(uuid_D);
  mpiBroadcastInt(nmode_D);
  mpiBroadcastInt64_tArray(idx_D, nmode_D);
  mpiBroadcastC64(alpha);
  mpiBroadcastC64(beta);

  //workers perform the tensor contraction
  if(rank == rootRank) waitWorkersFinished();
  return 0;
}

#endif
#endif

