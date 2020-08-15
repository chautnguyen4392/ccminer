#ifndef KECCAK_H
#define KEKKAC_H

extern "C" void prepare_keccak512(int thr_id, const uint32_t host_pdata[20]);
extern "C" void pre_keccak512(int thr_id, int stream, uint32_t nonce, int throughput, uint32_t keylen);
extern "C" void post_keccak512(int thr_id, int stream, uint32_t nonce, int throughput, uint32_t keylen);

#endif // #ifndef KEKKAC_H
