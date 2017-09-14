void xor_salsa8(__global uint * B, __global uint * Bx)
{
	uint x00,x01,x02,x03,x04,x05,x06,x07,x08,x09,x10,x11,x12,x13,x14,x15;
	int i;

	x00 = (B[ 0] ^= Bx[ 0]);
	x01 = (B[ 1] ^= Bx[ 1]);
	x02 = (B[ 2] ^= Bx[ 2]);
	x03 = (B[ 3] ^= Bx[ 3]);
	x04 = (B[ 4] ^= Bx[ 4]);
	x05 = (B[ 5] ^= Bx[ 5]);
	x06 = (B[ 6] ^= Bx[ 6]);
	x07 = (B[ 7] ^= Bx[ 7]);
	x08 = (B[ 8] ^= Bx[ 8]);
	x09 = (B[ 9] ^= Bx[ 9]);
	x10 = (B[10] ^= Bx[10]);
	x11 = (B[11] ^= Bx[11]);
	x12 = (B[12] ^= Bx[12]);
	x13 = (B[13] ^= Bx[13]);
	x14 = (B[14] ^= Bx[14]);
	x15 = (B[15] ^= Bx[15]);
	for (i = 0; i < 8; i += 2) {
#define R(a, b) (((a) << (b)) | ((a) >> (32 - (b))))
		/* Operate on columns. */
		x04 ^= R(x00+x12, 7);	x09 ^= R(x05+x01, 7);
		x14 ^= R(x10+x06, 7);	x03 ^= R(x15+x11, 7);
		
		x08 ^= R(x04+x00, 9);	x13 ^= R(x09+x05, 9);
		x02 ^= R(x14+x10, 9);	x07 ^= R(x03+x15, 9);
		
		x12 ^= R(x08+x04,13);	x01 ^= R(x13+x09,13);
		x06 ^= R(x02+x14,13);	x11 ^= R(x07+x03,13);
		
		x00 ^= R(x12+x08,18);	x05 ^= R(x01+x13,18);
		x10 ^= R(x06+x02,18);	x15 ^= R(x11+x07,18);
		
		/* Operate on rows. */
		x01 ^= R(x00+x03, 7);	x06 ^= R(x05+x04, 7);
		x11 ^= R(x10+x09, 7);	x12 ^= R(x15+x14, 7);
		
		x02 ^= R(x01+x00, 9);	x07 ^= R(x06+x05, 9);
		x08 ^= R(x11+x10, 9);	x13 ^= R(x12+x15, 9);
		
		x03 ^= R(x02+x01,13);	x04 ^= R(x07+x06,13);
		x09 ^= R(x08+x11,13);	x14 ^= R(x13+x12,13);
		
		x00 ^= R(x03+x02,18);	x05 ^= R(x04+x07,18);
		x10 ^= R(x09+x08,18);	x15 ^= R(x14+x13,18);
#undef R
	}
	B[ 0] += x00;
	B[ 1] += x01;
	B[ 2] += x02;
	B[ 3] += x03;
	B[ 4] += x04;
	B[ 5] += x05;
	B[ 6] += x06;
	B[ 7] += x07;
	B[ 8] += x08;
	B[ 9] += x09;
	B[10] += x10;
	B[11] += x11;
	B[12] += x12;
	B[13] += x13;
	B[14] += x14;
	B[15] += x15;
}


__kernel void test(__global uint * X, __global uint * V)
{
    size_t id = get_global_id(0) ;
    //printf("get_global_id %i, ", get_global_id(0)); // What is my current work item in the whole work group.
    //printf("get_global_size %i, ", get_global_size(0)); // what to total work items in this NDR is.
    //printf("get_global_offset %i, ", get_global_offset(0));
    //printf("get_group_id %i, ", get_group_id(0));
    //printf("get_local_id %i, ", get_local_id(0));
    //printf("get_local_size %i, ", get_local_size(0));
    //printf("get_num_groups %i\n", get_num_groups(0));
    
    /*
    __local int slow_down;
    //#pragma unroll
    for (int i = 0; i < 32; i++) {
        //#pragma unroll
        for (int k = 0; k < 4; k++) {
            //#pragma unroll
            //for (slow_down = 0; slow_down < 1000000; slow_down++) {
                X[id * 32 * 4 + k * 32 + i] = X[id * 32 * 4 + k * 32 + i] + 1;
            //}
        }
    }
    */
    __local uint i, j, k;
    size_t wi_id,gi_id;
    wi_id = get_local_id(0);
    gi_id = get_group_id(0);
    size_t wi_size = get_global_size(0);
    size_t thr_id = get_global_size(0) * get_global_offset(0);
    //printf("wi: %llu;\n", wi_id);
    //printf("wi: %llu;\n",wi_size);
    //printf("gi: %llu;\n", gi_id);
    //printf("offset: %i\n", get_global_offset(0));
    //printf("%llu\n", thr_id);
    ulong scratch_pad = (thr_id*32*1024)+wi_id*32*1024;
    ulong work_item = thr_id + wi_id * 32;
    int jmpe = 1; 
    for(i = 0; i < 1024; i=i+1){
        if(i % jmpe == 0) {
        for(j = 0; j < 32; ++j) // memcopy 
            V[scratch_pad + (i * 32) + j] = X[work_item + j];
        }
        xor_salsa8(&X[work_item], &X[work_item + 16]);
        xor_salsa8(&X[work_item + 16], &X[work_item]);
        
    }
    
    for(i = 0; i < 1024; ++i){
        j = 32 * (X[work_item + 16] & 0x3FF);
        for(k = 0; k < 32; ++k)
            X[work_item + k] = V[scratch_pad + j + k] ^ X[work_item + k];
        
        xor_salsa8(&X[work_item], &X[work_item + 16]);
        xor_salsa8(&X[work_item + 16], &X[work_item]);

    }
    

}

/*
// global * <--------local-------->
// thr_id * ((EXTRA_THROUGHPUT * 4) * 32
__kernel void scrypt_core(__global uint *X, __global uint *V)
{
    __local uint i, j, k;
    size_t lid, gid;
    lid = get_local_id(0);
    lindx = lid * 32;
    gid = get_global_id(0);
    
    for(i = 0; i < 1024; ++i){
        for(j = 0; j < 32; ++j) // memcopy
            V[i*32 + j] = x[j];
         
        //xor_salsa
        //xor_salsa
    }
    for(i = 0; i < 1024; ++i){
        j = 32 * (X[16] & 0x3FF);
        for(k = 0; k < 32; ++k)
            X[k] = V[j + k] ^ X[k];
        
        //xor_salsa8
        //xor_salsa8
    }
}
*/