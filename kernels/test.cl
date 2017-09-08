__kernel void test(__global int* X)
{
    size_t id = get_global_id(0) ;
    //printf("id: %i\n", id);
    for (int i = 0; i < 32; i++)
        for (int k = 0; k < 4; k++)
            X[id * 32 * 4 + k * 32 + i] = X[id * 32 * 4 + k * 32 + i] + 1;
}
