/*
 * scrypt-jane by Andrew M, https://github.com/floodyberry/scrypt-jane
 *
 * Public Domain or MIT License, whichever is easier
 *
 * Adapted to ccminer by tpruvot@github (2015)
 */

#include "miner.h"

#include "scrypt/scrypt-jane.h"
#include "scrypt/code/scrypt-jane-portable.h"
#include "scrypt/code/scrypt-jane-chacha.h"
#include "scrypt/keccak.h"

#include "scrypt/salsa_kernel.h"

#define scrypt_maxN 30  /* (1 << (30 + 1)) = ~2 billion */
#define scrypt_r_32kb 8 /* (1 << 8) = 256 * 2 blocks in a chunk * 64 bytes = Max of 32kb in a chunk */
#define scrypt_maxr scrypt_r_32kb /* 32kb */
#define scrypt_maxp 25  /* (1 << 25) = ~33 million */

// ---------------------------- BEGIN keccak functions ------------------------------------

#define SCRYPT_HASH "Keccak-512"
#define SCRYPT_HASH_DIGEST_SIZE 64
#define SCRYPT_KECCAK_F 1600
#define SCRYPT_KECCAK_C (SCRYPT_HASH_DIGEST_SIZE * 8 * 2) /* 1024 */
#define SCRYPT_KECCAK_R (SCRYPT_KECCAK_F - SCRYPT_KECCAK_C) /* 576 */
#define SCRYPT_HASH_BLOCK_SIZE (SCRYPT_KECCAK_R / 8)

typedef uint8_t scrypt_hash_digest[SCRYPT_HASH_DIGEST_SIZE];

typedef struct scrypt_hash_state_t {
	uint64_t state[SCRYPT_KECCAK_F / 64];
	uint32_t leftover;
	uint8_t buffer[SCRYPT_HASH_BLOCK_SIZE];
} scrypt_hash_state;

static const uint64_t keccak_round_constants[24] = {
	0x0000000000000001ull, 0x0000000000008082ull,
	0x800000000000808aull, 0x8000000080008000ull,
	0x000000000000808bull, 0x0000000080000001ull,
	0x8000000080008081ull, 0x8000000000008009ull,
	0x000000000000008aull, 0x0000000000000088ull,
	0x0000000080008009ull, 0x000000008000000aull,
	0x000000008000808bull, 0x800000000000008bull,
	0x8000000000008089ull, 0x8000000000008003ull,
	0x8000000000008002ull, 0x8000000000000080ull,
	0x000000000000800aull, 0x800000008000000aull,
	0x8000000080008081ull, 0x8000000000008080ull,
	0x0000000080000001ull, 0x8000000080008008ull
};

static void keccak_block(scrypt_hash_state *S, const uint8_t *in)
{
	size_t i;
	uint64_t *s = S->state, t[5], u[5], v, w;

	/* absorb input */
	for (i = 0; i < SCRYPT_HASH_BLOCK_SIZE / 8; i++, in += 8)
		s[i] ^= U8TO64_LE(in);

	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ ROTL64(t[1], 1);
		u[1] = t[0] ^ ROTL64(t[2], 1);
		u[2] = t[1] ^ ROTL64(t[3], 1);
		u[3] = t[2] ^ ROTL64(t[4], 1);
		u[4] = t[3] ^ ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[ 1];
		s[ 1] = ROTL64(s[ 6], 44);
		s[ 6] = ROTL64(s[ 9], 20);
		s[ 9] = ROTL64(s[22], 61);
		s[22] = ROTL64(s[14], 39);
		s[14] = ROTL64(s[20], 18);
		s[20] = ROTL64(s[ 2], 62);
		s[ 2] = ROTL64(s[12], 43);
		s[12] = ROTL64(s[13], 25);
		s[13] = ROTL64(s[19],  8);
		s[19] = ROTL64(s[23], 56);
		s[23] = ROTL64(s[15], 41);
		s[15] = ROTL64(s[ 4], 27);
		s[ 4] = ROTL64(s[24], 14);
		s[24] = ROTL64(s[21],  2);
		s[21] = ROTL64(s[ 8], 55);
		s[ 8] = ROTL64(s[16], 45);
		s[16] = ROTL64(s[ 5], 36);
		s[ 5] = ROTL64(s[ 3], 28);
		s[ 3] = ROTL64(s[18], 21);
		s[18] = ROTL64(s[17], 15);
		s[17] = ROTL64(s[11], 10);
		s[11] = ROTL64(s[ 7],  6);
		s[ 7] = ROTL64(s[10],  3);
		s[10] = ROTL64(    v,  1);

		/* chi: a[i,j] ^= ~b[i,j+1] & b[i,j+2] */
		v = s[ 0]; w = s[ 1]; s[ 0] ^= (~w) & s[ 2]; s[ 1] ^= (~s[ 2]) & s[ 3]; s[ 2] ^= (~s[ 3]) & s[ 4]; s[ 3] ^= (~s[ 4]) & v; s[ 4] ^= (~v) & w;
		v = s[ 5]; w = s[ 6]; s[ 5] ^= (~w) & s[ 7]; s[ 6] ^= (~s[ 7]) & s[ 8]; s[ 7] ^= (~s[ 8]) & s[ 9]; s[ 8] ^= (~s[ 9]) & v; s[ 9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		/* iota: a[0,0] ^= round constant */
		s[0] ^= keccak_round_constants[i];
	}
}

static void scrypt_hash_init(scrypt_hash_state *S) {
	memset(S, 0, sizeof(*S));
}

static void scrypt_hash_update(scrypt_hash_state *S, const uint8_t *in, size_t inlen)
{
	size_t want;

	/* handle the previous data */
	if (S->leftover) {
		want = (SCRYPT_HASH_BLOCK_SIZE - S->leftover);
		want = (want < inlen) ? want : inlen;
		memcpy(S->buffer + S->leftover, in, want);
		S->leftover += (uint32_t)want;
		if (S->leftover < SCRYPT_HASH_BLOCK_SIZE)
			return;
		in += want;
		inlen -= want;
		keccak_block(S, S->buffer);
	}

	/* handle the current data */
	while (inlen >= SCRYPT_HASH_BLOCK_SIZE) {
		keccak_block(S, in);
		in += SCRYPT_HASH_BLOCK_SIZE;
		inlen -= SCRYPT_HASH_BLOCK_SIZE;
	}

	/* handle leftover data */
	S->leftover = (uint32_t)inlen;
	if (S->leftover)
		memcpy(S->buffer, in, S->leftover);
}

static void scrypt_hash_finish(scrypt_hash_state *S, uint8_t *hash)
{
	size_t i;

	S->buffer[S->leftover] = 0x01;
	memset(S->buffer + (S->leftover + 1), 0, SCRYPT_HASH_BLOCK_SIZE - (S->leftover + 1));
	S->buffer[SCRYPT_HASH_BLOCK_SIZE - 1] |= 0x80;
	keccak_block(S, S->buffer);

	for (i = 0; i < SCRYPT_HASH_DIGEST_SIZE; i += 8) {
		U64TO8_LE(&hash[i], S->state[i / 8]);
	}
}

// ---------------------------- END keccak functions ------------------------------------

// ---------------------------- BEGIN PBKDF2 functions ------------------------------------

typedef struct scrypt_hmac_state_t {
	scrypt_hash_state inner, outer;
} scrypt_hmac_state;


static void scrypt_hash(scrypt_hash_digest hash, const uint8_t *m, size_t mlen)
{
	scrypt_hash_state st;

	scrypt_hash_init(&st);
	scrypt_hash_update(&st, m, mlen);
	scrypt_hash_finish(&st, hash);
}

/* hmac */
static void scrypt_hmac_init(scrypt_hmac_state *st, const uint8_t *key, size_t keylen)
{
	uint8_t pad[SCRYPT_HASH_BLOCK_SIZE] = {0}; // SCRYPT_HASH_BLOCK_SIZE = 72
	size_t i;

	scrypt_hash_init(&st->inner);
	scrypt_hash_init(&st->outer);

	if (keylen <= SCRYPT_HASH_BLOCK_SIZE) {
		/* use the key directly if it's <= blocksize bytes */
		memcpy(pad, key, keylen);
	} else {
		/* if it's > blocksize bytes, hash it */
		scrypt_hash(pad, key, keylen);
	}

	/* inner = (key ^ 0x36) */
	/* h(inner || ...) */
	for (i = 0; i < SCRYPT_HASH_BLOCK_SIZE; i++)
		pad[i] ^= 0x36;
	scrypt_hash_update(&st->inner, pad, SCRYPT_HASH_BLOCK_SIZE);

	/* outer = (key ^ 0x5c) */
	/* h(outer || ...) */
	for (i = 0; i < SCRYPT_HASH_BLOCK_SIZE; i++)
		pad[i] ^= (0x5c ^ 0x36);
	scrypt_hash_update(&st->outer, pad, SCRYPT_HASH_BLOCK_SIZE);
}

static void scrypt_hmac_update(scrypt_hmac_state *st, const uint8_t *m, size_t mlen)
{
	/* h(inner || m...) */
	scrypt_hash_update(&st->inner, m, mlen);
}

static void scrypt_hmac_finish(scrypt_hmac_state *st, scrypt_hash_digest mac)
{
	/* h(inner || m) */
	scrypt_hash_digest innerhash;
	scrypt_hash_finish(&st->inner, innerhash);

	/* h(outer || h(inner || m)) */
	scrypt_hash_update(&st->outer, innerhash, sizeof(innerhash));
	scrypt_hash_finish(&st->outer, mac);
}

/*
 * Special version where N = 1
 *  - mikaelh
 */
void scrypt_pbkdf2_1(const uint8_t *password, size_t password_len,
	const uint8_t *salt, size_t salt_len, uint8_t *out, uint64_t bytes)
{
	scrypt_hmac_state hmac_pw, hmac_pw_salt, work;
	scrypt_hash_digest ti, u;
	uint8_t be[4];
	uint32_t i, blocks;

	/* bytes must be <= (0xffffffff - (SCRYPT_HASH_DIGEST_SIZE - 1)), which they will always be under scrypt */

	/* hmac(password, ...) */
	scrypt_hmac_init(&hmac_pw, password, password_len);

	/* hmac(password, salt...) */
	hmac_pw_salt = hmac_pw;
	scrypt_hmac_update(&hmac_pw_salt, salt, salt_len);

	blocks = ((uint32_t)bytes + (SCRYPT_HASH_DIGEST_SIZE - 1)) / SCRYPT_HASH_DIGEST_SIZE;
	for (i = 1; i <= blocks; i++) {
		/* U1 = hmac(password, salt || be(i)) */
		U32TO8_BE(be, i);
		work = hmac_pw_salt;
		scrypt_hmac_update(&work, be, 4);
		scrypt_hmac_finish(&work, ti);
		memcpy(u, ti, sizeof(u));

		memcpy(out, ti, (size_t) (bytes > SCRYPT_HASH_DIGEST_SIZE ? SCRYPT_HASH_DIGEST_SIZE : bytes));
		out += SCRYPT_HASH_DIGEST_SIZE;
		bytes -= SCRYPT_HASH_DIGEST_SIZE;
	}
}

// ---------------------------- END PBKDF2 functions ------------------------------------

static void scrypt_fatal_error_default(const char *msg) {
	fprintf(stderr, "%s\n", msg);
	exit(1);
}

static scrypt_fatal_errorfn scrypt_fatal_error = scrypt_fatal_error_default;

void scrypt_set_fatal_error_default(scrypt_fatal_errorfn fn) {
	scrypt_fatal_error = fn;
}

typedef struct scrypt_aligned_alloc_t {
	uint8_t *mem, *ptr;
} scrypt_aligned_alloc;

#if defined(SCRYPT_TEST_SPEED)
static uint8_t *mem_base = (uint8_t *)0;
static size_t mem_bump = 0;

/* allocations are assumed to be multiples of 64 bytes and total allocations not to exceed ~1.01gb */
static scrypt_aligned_alloc scrypt_alloc(uint64_t size)
{
	scrypt_aligned_alloc aa;
	if (!mem_base) {
		mem_base = (uint8_t *)malloc((1024 * 1024 * 1024) + (1024 * 1024) + (SCRYPT_BLOCK_BYTES - 1));
		if (!mem_base)
			scrypt_fatal_error("scrypt: out of memory");
		mem_base = (uint8_t *)(((size_t)mem_base + (SCRYPT_BLOCK_BYTES - 1)) & ~(SCRYPT_BLOCK_BYTES - 1));
	}
	aa.mem = mem_base + mem_bump;
	aa.ptr = aa.mem;
	mem_bump += (size_t)size;
	return aa;
}

static void scrypt_free(scrypt_aligned_alloc *aa)
{
	mem_bump = 0;
}
#else
static scrypt_aligned_alloc scrypt_alloc(uint64_t size)
{
	static const size_t max_alloc = (size_t)-1;
	scrypt_aligned_alloc aa;
	size += (SCRYPT_BLOCK_BYTES - 1);
	if (size > max_alloc)
		scrypt_fatal_error("scrypt: not enough address space on this CPU to allocate required memory");
	aa.mem = (uint8_t *)malloc((size_t)size);
	aa.ptr = (uint8_t *)(((size_t)aa.mem + (SCRYPT_BLOCK_BYTES - 1)) & ~(SCRYPT_BLOCK_BYTES - 1));
	if (!aa.mem)
		scrypt_fatal_error("scrypt: out of memory");
	return aa;
}

static void scrypt_free(scrypt_aligned_alloc *aa)
{
	free(aa->mem);
}
#endif


// yacoin: increasing Nfactor gradually
unsigned char GetNfactor(unsigned int nTimestamp)
{
	int l = 0;

	unsigned int Nfactor = 0;

	// Yacoin defaults
	unsigned int Ntimestamp = 1367991200;
	unsigned int minN = 4;
	unsigned int maxN = 30;

	if (strlen(jane_params) > 0) {
		if (!strcmp(jane_params, "YAC") || !strcasecmp(jane_params, "Yacoin")) {} // No-Op
		//
		// NO WARRANTY FOR CORRECTNESS. Look for the int64 nChainStartTime constant
		// in the src/main.cpp file of the official wallet clients as well as the
		// const unsigned char minNfactor and const unsigned char maxNfactor
		//
		else if (!strcmp(jane_params, "YBC") || !strcasecmp(jane_params, "YBCoin")) {
			// YBCoin:   1372386273, minN:  4, maxN: 30
			Ntimestamp = 1372386273; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "ZZC") || !strcasecmp(jane_params, "ZZCoin")) {
			// ZcCoin:   1375817223, minN: 12, maxN: 30
			Ntimestamp = 1375817223; minN= 12; maxN= 30;
		} else if (!strcmp(jane_params, "FEC") || !strcasecmp(jane_params, "FreeCoin")) {
			// FreeCoin: 1375801200, minN:  6, maxN: 32
			Ntimestamp = 1375801200; minN=  6; maxN= 32;
		} else if (!strcmp(jane_params, "ONC") || !strcasecmp(jane_params, "OneCoin")) {
			// OneCoin:  1371119462, minN:  6, maxN: 30
			Ntimestamp = 1371119462; minN=  6; maxN= 30;
		} else if (!strcmp(jane_params, "QQC") || !strcasecmp(jane_params, "QQCoin")) {
			// QQCoin:   1387769316, minN:  4, maxN: 30
			Ntimestamp = 1387769316; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "GPL") || !strcasecmp(jane_params, "GoldPressedLatinum")) {
			// GoldPressedLatinum:1377557832, minN:  4, maxN: 30
			Ntimestamp = 1377557832; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "MRC") || !strcasecmp(jane_params, "MicroCoin")) {
			// MicroCoin:1389028879, minN:  4, maxN: 30
			Ntimestamp = 1389028879; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "APC") || !strcasecmp(jane_params, "AppleCoin")) {
			// AppleCoin:1384720832, minN:  4, maxN: 30
			Ntimestamp = 1384720832; minN=  4; maxN= 30;
		} else if (!strcmp(jane_params, "CPR") || !strcasecmp(jane_params, "Copperbars")) {
			// Copperbars:1376184687, minN: 4, maxN: 30
			Ntimestamp = 1376184687; minN= 4; maxN= 30;
		} else if (!strcmp(jane_params, "CACH") || !strcasecmp(jane_params, "CacheCoin")) {
			// CacheCoin:1388949883, minN: 4, maxN: 30
			Ntimestamp = 1388949883; minN= 4; maxN= 30;
		} else if (!strcmp(jane_params, "UTC") || !strcasecmp(jane_params, "UltraCoin")) {
			// MicroCoin:1388361600, minN: 4, maxN: 30
			Ntimestamp = 1388361600; minN= 4; maxN= 30;
		} else if (!strcmp(jane_params, "VEL") || !strcasecmp(jane_params, "VelocityCoin")) {
			// VelocityCoin:1387769316, minN: 4, maxN: 30
			Ntimestamp = 1387769316; minN= 4; maxN= 30;
		} else if (!strcmp(jane_params, "ITC") || !strcasecmp(jane_params, "InternetCoin")) {
			// InternetCoin:1388385602, minN: 4, maxN: 30
			Ntimestamp = 1388385602; minN= 4; maxN= 30;
		} else if (!strcmp(jane_params, "RAD") || !strcasecmp(jane_params, "RadioactiveCoin")) {
			// InternetCoin:1389196388, minN: 4, maxN: 30
			Ntimestamp = 1389196388; minN= 4; maxN= 30;
		} else if (!strcmp(jane_params, "LEO") || !strcasecmp(jane_params, "LEOCoin")) {
			// LEOCoin:1402845776, minN: 4, maxN: 30
			Ntimestamp = 1402845776; minN= 4; maxN= 30;
		} else {
			if (sscanf(jane_params, "%u,%u,%u", &Ntimestamp, &minN, &maxN) != 3)
			if (sscanf(jane_params, "%u", &Nfactor) == 1) return Nfactor; // skip bounding against minN, maxN
			else applog(LOG_INFO, "Unable to parse scrypt-jane parameters: '%s'. Defaulting to Yacoin.", jane_params);
		}
	}
	// determination based on the constants determined above
	if (nTimestamp <= Ntimestamp)
		return minN;

	unsigned long int s = nTimestamp - Ntimestamp;
	while ((s >> 1) > 3) {
		l += 1;
		s >>= 1;
	}

	s &= 3;

	int n = (l * 170 + s * 25 - 2320) / 100;

	if (n < 0) n = 0;

	if (n > 255)
		printf("GetNfactor(%d) - something wrong(n == %d)\n", nTimestamp, n);

	Nfactor = n;
	if (Nfactor<minN) return minN;
	if (Nfactor>maxN) return maxN;
	return Nfactor;
}

static bool init[MAX_GPUS] = { 0 };

// cleanup
void free_scrypt_jane(int thr_id)
{
	int dev_id = device_map[thr_id];

	if (!init[thr_id])
		return;

	cudaSetDevice(dev_id);
	cudaDeviceSynchronize();
	cudaDeviceReset(); // well, simple way to free ;)

	init[thr_id] = false;
}

#define bswap_32x4(x) ((((x) << 24) & 0xff000000u) | (((x) << 8) & 0x00ff0000u) \
					 | (((x) >> 8) & 0x0000ff00u) | (((x) >> 24) & 0x000000ffu))
#define bswap_64x16(val) \
 ( (((val) >> 56) & 0x00000000000000ffu) | (((val) >> 40) & 0x000000000000ff00u) | \
   (((val) >> 24) & 0x0000000000ff0000u) | (((val) >>  8) & 0x00000000ff000000u) | \
   (((val) <<  8) & 0x000000ff00000000u) | (((val) << 24) & 0x0000ff0000000000u) | \
   (((val) << 40) & 0x00ff000000000000u) | (((val) << 56) & 0xff00000000000000u) )

static int s_Nfactor = 0;

int scanhash_scrypt_jane(int thr_id, struct work *work, uint32_t max_nonce, unsigned long *hashes_done,
	unsigned char *scratchbuf, struct timeval *tv_start, struct timeval *tv_end, int block_version)
{
	uint32_t *pdata = work->data;
	// GET HASH TARGET
	uint32_t *ptarget = work->target;
	const uint32_t Htarg = ptarget[7];
	uint32_t N;

	/* BEGIN print received data */
    if (opt_debug)
    {
		if (block_version < 7)
		{
			#pragma pack(push, 1)
			struct block_header
			{
				unsigned int version;
				uint32_t prev_block[8];
				uint32_t merkle_root[8];
				uint32_t timestamp;
				unsigned int bits;
				unsigned int nonce;

			};
			#pragma pack(pop)

			struct block_header pTempData;
			memcpy((void*)&pTempData, (const void*)work->data, sizeof(pTempData));
			// Byte reverse
			for (unsigned int i = 0; i < sizeof(pTempData)/sizeof( uint32_t ); ++i)
		  //for (int i = 0; i < 128/4; i++) //really, the limit is sizeof( *pdata ) / sizeof( uint32_t
				((uint32_t *)&pTempData)[i] = bswap_32x4(((uint32_t *)&pTempData)[i]);

			char *hashPrevBlock_str = get_target_string(pTempData.prev_block);
			char *hashMerkleRoot_str = get_target_string(pTempData.merkle_root);
			applog(LOG_DEBUG,
					"TACA => scanhash_scrypt_jane[%d], received block header data,\n"
							"pTempData->nVersion = %d,\n"
							"pTempData->hashPrevBlock = %s,\n"
							"pTempData->hashMerkleRoot = %s,\n"
							"pTempData->nTime = %lld,\n"
							"pTempData->nBits = %u,\n"
							"pTempData->nNonce = %u\n", thr_id,
					pTempData.version, hashPrevBlock_str, hashMerkleRoot_str,
					pTempData.timestamp, pTempData.bits, pTempData.nonce);
			free(hashPrevBlock_str);
			free(hashMerkleRoot_str);
		}
		else
		{
			/* BEGIN print received data */
			#pragma pack(push, 1)
			struct block_header
			{
				unsigned int version;
				uint32_t prev_block[8];
				uint32_t merkle_root[8];
				::int64_t timestamp;
				unsigned int bits;
				unsigned int nonce;

			};
			#pragma pack(pop)

			struct block_header pTempData;
			memcpy((void*)&pTempData, (const void*)work->data, sizeof(pTempData));
			// Byte reverse
			for (unsigned int i = 0; i < sizeof(pTempData)/sizeof( uint32_t ); ++i)
		  //for (int i = 0; i < 128/4; i++) //really, the limit is sizeof( *pdata ) / sizeof( uint32_t
				((uint32_t *)&pTempData)[i] = bswap_32x4(((uint32_t *)&pTempData)[i]);

			char *hashPrevBlock_str = get_target_string(pTempData.prev_block);
			char *hashMerkleRoot_str = get_target_string(pTempData.merkle_root);
			applog(LOG_DEBUG,
					"TACA => scanhash_scrypt_jane[%d], received block header data,\n"
							"pTempData->nVersion = %d,\n"
							"pTempData->hashPrevBlock = %s,\n"
							"pTempData->hashMerkleRoot = %s,\n"
							"pTempData->nTime = %lld,\n"
							"pTempData->nBits = %u,\n"
							"pTempData->nNonce = %u\n", thr_id,
					pTempData.version, hashPrevBlock_str, hashMerkleRoot_str,
					pTempData.timestamp, pTempData.bits, pTempData.nonce);
			free(hashPrevBlock_str);
			free(hashMerkleRoot_str);
		}
    }
    /* END print received data */

	if (s_Nfactor == 0 && strlen(jane_params) > 0)
		applog(LOG_INFO, "Given scrypt-jane parameters: %s", jane_params);

	// Default value is from after hardfork block case
	int Nfactor = 21; // Nfactor is fixed after hardfork
	int block_header_size = 84;
	if (block_version < 7)
	{
		// Get nFactor based on block version
		Nfactor = GetNfactor(bswap_32x4(pdata[17]));
		block_header_size = 80;
	}

	if (Nfactor > scrypt_maxN) {
		scrypt_fatal_error("scrypt: N out of range");
	}
	N = (1 << (Nfactor + 1));

	if (Nfactor != s_Nfactor)
	{
		opt_nfactor = Nfactor;
		applog(LOG_INFO, "N-factor is %d (%d)!", Nfactor, N);
		if (s_Nfactor != 0) {
			// handle N-factor increase at runtime
			// by adjusting the lookup_gap by factor 2
			if (s_Nfactor == Nfactor-1)
				for (int i=0; i < 8; ++i)
					device_lookup_gap[i] *= 2;
		}
		s_Nfactor = Nfactor;
	}



	static __thread int throughput = 0;
	if(!init[thr_id]) {
		int dev_id = device_map[thr_id];

		cudaSetDevice(dev_id);
		cudaDeviceSynchronize();
		cudaDeviceReset();
		cudaSetDevice(dev_id);

		throughput = cuda_throughput(thr_id);
		gpulog(LOG_INFO, thr_id, "Intensity set to %g, %u cuda threads", throughput2intensity(throughput), throughput);

		init[thr_id] = true;
	}

	if(throughput == 0)
		return -1;

	gettimeofday(tv_start, NULL);

	uint32_t *data[2] = { new uint32_t[(block_header_size/4)*throughput], new uint32_t[(block_header_size/4)*throughput] };
	uint32_t* hash[2]   = { cuda_hashbuffer(thr_id,0), cuda_hashbuffer(thr_id,1) };

	uint32_t n = pdata[(block_header_size/4 - 1)];

    if (opt_debug)
    {
    	uint32_t nNonce = bswap_32x4(pdata[(block_header_size/4 - 1)]);
		char *target_str = get_target_string(ptarget);
		applog(LOG_DEBUG,
				"TACA => scanhash_scrypt_jane[%d], Nfactor = %d, target = %s, Htarg = %x, throughput = %d, parallel = %d, nNonce = %u",
				thr_id, Nfactor, target_str, Htarg, throughput, parallel,
				nNonce);
		free(target_str);
    }

	/* byte swap pdata into data[0]/[1] arrays */
	for (int k=0; k<2; ++k) {
		for(int z=0;z<(block_header_size/4);z++) data[k][z] = bswap_32x4(pdata[z]);
		for(int i=1;i<throughput;++i) memcpy(&data[k][(block_header_size/4)*i], &data[k][0], (block_header_size/4)*sizeof(uint32_t));
	}
	if (parallel == 2) prepare_keccak512(thr_id, pdata, block_header_size);

	scrypt_aligned_alloc Xbuf[2] = { scrypt_alloc(128 * throughput), scrypt_alloc(128 * throughput) };
	scrypt_aligned_alloc Vbuf = scrypt_alloc(N * 128);
	scrypt_aligned_alloc Ybuf = scrypt_alloc(128);

	uint32_t nonce[2];
	uint32_t* cuda_X[2]      = { cuda_transferbuffer(thr_id,0), cuda_transferbuffer(thr_id,1) };

#if !defined(SCRYPT_CHOOSE_COMPILETIME)
	scrypt_ROMixfn scrypt_ROMix = scrypt_getROMix();
#endif

	int cur = 0, nxt = 1;
	int iteration = 0;

	do {
		nonce[nxt] = n;

		if (parallel < 2)
		{
			// half of cpu

			for(int i=0;i<throughput;++i) {
				uint32_t tmp_nonce = n++;
				data[nxt][(block_header_size/4)*i + (block_header_size/4 - 1)] = bswap_32x4(tmp_nonce);
			}

			for(int i=0;i<throughput;++i)
				scrypt_pbkdf2_1(
						(unsigned char*) &data[nxt][(block_header_size / 4) * i],
						block_header_size,
						(unsigned char*) &data[nxt][(block_header_size / 4) * i],
						block_header_size, Xbuf[nxt].ptr + 128 * i, 128);

			memcpy(cuda_X[nxt], Xbuf[nxt].ptr, 128 * throughput);
			cuda_scrypt_serialize(thr_id, nxt);
			cuda_scrypt_HtoD(thr_id, cuda_X[nxt], nxt);
			cuda_scrypt_core(thr_id, nxt, N);
			cuda_scrypt_done(thr_id, nxt);

			cuda_scrypt_DtoH(thr_id, cuda_X[nxt], nxt, false);

			//cuda_scrypt_flush(thr_id, nxt);
			if(!cuda_scrypt_sync(thr_id, nxt)) {
				break;
			}

			memcpy(Xbuf[cur].ptr, cuda_X[cur], 128 * throughput);
			for(int i=0;i<throughput;++i)
				scrypt_pbkdf2_1(
						(unsigned char*) &data[cur][(block_header_size / 4) * i],
						block_header_size, Xbuf[cur].ptr + 128 * i, 128,
						(unsigned char*) (&hash[cur][8 * i]), 32);

#define VERIFY_ALL 0
#if VERIFY_ALL
			{
				/* 2: X = ROMix(X) */
				for(int i=0;i<throughput;++i)
					scrypt_ROMix_1((scrypt_mix_word_t *)(Xbuf[cur].ptr + 128 * i), (scrypt_mix_word_t *)Ybuf.ptr, (scrypt_mix_word_t *)Vbuf.ptr, N);

				unsigned int err = 0;
				for(int i=0;i<throughput;++i) {
					unsigned char *ref = (Xbuf[cur].ptr + 128 * i);
					unsigned char *dat = (unsigned char*)(cuda_X[cur] + 32 * i);
					if (memcmp(ref, dat, 128) != 0)
					{
						err++;
#if 0
						uint32_t *ref32 = (uint32_t*) ref;
						uint32_t *dat32 = (uint32_t*) dat;
						for (int j=0; j<32; ++j) {
							if (ref32[j] != dat32[j])
							fprintf(stderr, "ref32[i=%d][j=%d] = $%08x / $%08x\n", i, j, ref32[j], dat32[j]);
						}
#endif
					}
				}
				if (err > 0) fprintf(stderr, "%d out of %d hashes differ.\n", err, throughput);
			}
#endif
		} else {

			// all on gpu

			n += throughput;
			if (opt_debug && (iteration % 64 == 0))
				applog(LOG_DEBUG, "GPU #%d: n=%x", device_map[thr_id], n);

		    if (opt_debug)
		    {
				for (int i = 0; iteration > 0 && i < throughput; i++) {
					char *target_str = get_target_string(ptarget);
					char *hash_cur_str = get_target_string(&hash[cur][8 * i]);
					char *hash_nxt_str = get_target_string(&hash[nxt][8 * i]);

					applog(LOG_DEBUG,
							"TACA => scanhash_scrypt_jane[%d], BEFORE scan hash, i = %d, hash[cur][8*i] = %s, hash[nxt][8*i] = %s, Htarg = %x, nonce[cur] = %u, nonce[nxt] = %u",
							thr_id, i, hash_cur_str, hash_nxt_str, Htarg,
							nonce[cur], nonce[nxt]);
					free(target_str);
					free(hash_cur_str);
					free(hash_nxt_str);
				}
		    }

			cuda_scrypt_serialize(thr_id, nxt);
			pre_keccak512(thr_id, nxt, nonce[nxt], throughput, block_header_size);
			cuda_scrypt_core(thr_id, nxt, N);
			//cuda_scrypt_flush(thr_id, nxt);
			if (!cuda_scrypt_sync(thr_id, nxt)) {
				break;
			}

			post_keccak512(thr_id, nxt, nonce[nxt], throughput, block_header_size);
			cuda_scrypt_done(thr_id, nxt);

			cuda_scrypt_DtoH(thr_id, hash[nxt], nxt, true);

		    if (opt_debug)
		    {
				for (int i=0; iteration > 0 && i<throughput; i++)
				{
					char *target_str = get_target_string(ptarget);
					char *hash_cur_str = get_target_string(&hash[cur][8 * i]);
					char *hash_nxt_str = get_target_string(&hash[nxt][8 * i]);

					applog(LOG_DEBUG,
							"TACA => scanhash_scrypt_jane[%d], AFTER scan hash, i = %d, hash[cur][8*i] = %s, hash[nxt][8*i] = %s, Htarg = %x, nonce[cur] = %u, nonce[nxt] = %u",
							thr_id, i, hash_cur_str, hash_nxt_str, Htarg,
							nonce[cur], nonce[nxt]);
					free(target_str);
					free(hash_cur_str);
					free(hash_nxt_str);
				}
		    }
			//cuda_scrypt_flush(thr_id, nxt); // made by cuda_scrypt_sync
			if (!cuda_scrypt_sync(thr_id, nxt)) {
				break;
			}
		}

		for (int i=0; iteration > 0 && i<throughput; i++)
		{
			if (hash[cur][8*i+7] <= Htarg && fulltest(&hash[cur][8*i], ptarget))
			{
				uint32_t _ALIGN(64) thash[8], tdata[(block_header_size / 4)];
				uint32_t tmp_nonce = nonce[cur] + i;

				for(int z=0;z<(block_header_size / 4 - 1);z++)
					tdata[z] = bswap_32x4(pdata[z]);
				tdata[(block_header_size / 4 - 1)] = bswap_32x4(tmp_nonce);

				scrypt_pbkdf2_1((unsigned char *)tdata, block_header_size, (unsigned char *)tdata, block_header_size, Xbuf[cur].ptr + 128 * i, 128);
				scrypt_ROMix_1((scrypt_mix_word_t *)(Xbuf[cur].ptr + 128 * i), (scrypt_mix_word_t *)(Ybuf.ptr), (scrypt_mix_word_t *)(Vbuf.ptr), N);
				scrypt_pbkdf2_1((unsigned char *)tdata, block_header_size, Xbuf[cur].ptr + 128 * i, 128, (unsigned char *)thash, 32);

				char *hash_cpu_str = get_target_string(thash);
				char *hash_gpu_str = get_target_string(&hash[cur][8*i]);
				applog(LOG_NOTICE,
						"TACA => scanhash_scrypt_jane[%d], FOUND a possible solution at i = %d with nonce = %u, hash_cpu_str = %s, hash_gpu_str = %s",
						thr_id, i, tmp_nonce, hash_cpu_str, hash_gpu_str);

				if (memcmp(thash, &hash[cur][8*i], 32) == 0)
				{
					applog(LOG_NOTICE,
							"TACA => scanhash_scrypt_jane[%d], FOUND a solution at i = %d with nonce = %u, hash_cpu_str = %s, hash_gpu_str = %s",
							thr_id, i, tmp_nonce, hash_cpu_str, hash_gpu_str);
					free(hash_cpu_str);
					free(hash_gpu_str);
					work_set_target_ratio(work, thash);
					*hashes_done = n - pdata[(block_header_size / 4 - 1)];
					pdata[(block_header_size / 4 - 1)] = tmp_nonce;
					scrypt_free(&Vbuf);
					scrypt_free(&Ybuf);
					scrypt_free(&Xbuf[0]); scrypt_free(&Xbuf[1]);
					delete[] data[0]; delete[] data[1];
					gettimeofday(tv_end, NULL);
					return 1;
				} else {
					free(hash_cpu_str);
					free(hash_gpu_str);
					applog(LOG_ERR,
							"TACA => scanhash_scrypt_jane[%d], result does not validate on CPU",
							thr_id);
					gpulog(LOG_WARNING, thr_id, "result does not validate on CPU! (i=%d, s=%d)", i, cur);
				}
			}
		}

		cur = (cur+1)&1;
		nxt = (nxt+1)&1;
		++iteration;
	} while (n <= max_nonce && !work_restart[thr_id].restart);

	scrypt_free(&Vbuf);
	scrypt_free(&Ybuf);
	scrypt_free(&Xbuf[0]); scrypt_free(&Xbuf[1]);
	delete[] data[0]; delete[] data[1];

	*hashes_done = n - pdata[(block_header_size / 4 - 1)];
	pdata[(block_header_size / 4 - 1)] = n;
	gettimeofday(tv_end, NULL);
	return 0;
}


static void scrypt_jane_hash_1_1(const uchar *password, size_t password_len, const uchar*salt, size_t salt_len, uint32_t N,
	uchar *out, uint32_t bytes, uint8_t *X, uint8_t *Y, uint8_t *V)
{
	uint32_t chunk_bytes, i;
	const uint32_t p = SCRYPT_P;

#if !defined(SCRYPT_CHOOSE_COMPILETIME)
	scrypt_ROMixfn scrypt_ROMix = scrypt_getROMix();
#endif

	chunk_bytes = SCRYPT_BLOCK_BYTES * SCRYPT_R * 2;

	/* 1: X = PBKDF2(password, salt) */
	scrypt_pbkdf2_1(password, password_len, salt, salt_len, X, chunk_bytes * p);

	/* 2: X = ROMix(X) */
	for (i = 0; i < p; i++)
		scrypt_ROMix_1((scrypt_mix_word_t *)(X + (chunk_bytes * i)), (scrypt_mix_word_t *)Y, (scrypt_mix_word_t *)V, N);

	/* 3: Out = PBKDF2(password, X) */
	scrypt_pbkdf2_1(password, password_len, X, chunk_bytes * p, out, (size_t) bytes);

#ifdef SCRYPT_PREVENT_STATE_LEAK
	/* This is an unnecessary security feature - mikaelh */
	scrypt_ensure_zero(Y, (p + 1) * chunk_bytes);
#endif
}

/* for cpu hash test */
void scryptjane_hash(void* output, const void* input)
{
	uint32_t Nsize = 1UL << (opt_nfactor + 1);
	uint64_t chunk_bytes;
	uint8_t *X, *Y;
	scrypt_aligned_alloc YX, V;

	chunk_bytes = 2ULL * SCRYPT_BLOCK_BYTES * SCRYPT_R;
	V  = scrypt_alloc(Nsize * chunk_bytes);
	YX = scrypt_alloc((SCRYPT_P + 1) * chunk_bytes);

	memset(V.ptr, 0, (size_t) (Nsize * chunk_bytes));

	Y = YX.ptr;
	X = Y + chunk_bytes;

	scrypt_jane_hash_1_1((uchar*)input, 80, (uchar*)input, 80, (uint32_t) Nsize, (uchar*)output, 32, X, Y, V.ptr);

	scrypt_free(&V);
	scrypt_free(&YX);
}
