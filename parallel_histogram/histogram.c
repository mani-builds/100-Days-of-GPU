#include <stdio.h>
#include <stdlib.h> // Add this for malloc

void histogram(char *data, unsigned int length, unsigned int *histo) {
  for (int i = 0; i < length; i++) {
    int alphabet_position = data[i] - 'a';
    if (alphabet_position >=0 && alphabet_position < 26)
    histo[alphabet_position/4]++;
  }
}

int main() {
  char *data= "programming massively parallel processors";
  unsigned int len = 41; // length of the chars
  unsigned int *histo;
  histo = (unsigned int *)malloc(7*sizeof(unsigned int));

  printf("Printing bins: \n");
  histogram(data, len,histo);
  for(int i=0; i<7; i++)
    printf("%d\t", histo[i]);
  printf("\n");

  return 0;

}
