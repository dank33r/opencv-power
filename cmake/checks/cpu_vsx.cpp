#include <stdio.h>
#include <altivec.h>
#define CV_VSX 1

#if defined CV_VSX
int test()
{
    int a[] = {0,1,2,3};
	vector int v;
    v = vec_xl( 0, a);
	return vec_extract(v, 0);
}
#else
#error "VSX is not supported"
#endif

int main() {
    printf("%d\n", test());
	return 0;
}
