#include <stdio.h>

int main(){
  int k = 0;
  int m = -10;
  for(int i=0; i<10; i++) k+=i;
  m += k;
  printf("%d, %d", k, m);
}
