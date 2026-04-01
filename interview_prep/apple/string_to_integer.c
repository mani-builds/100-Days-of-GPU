#include <stdio.h>
#include <stdbool.h>
#include <limits.h>

int myAtoi(char *s) {
    int i = 0, sign = 1;
    long res = 0; // Use long to detect overflow easily

    // 1. Skip leading whitespace
    while (s[i] == ' ') i++;

    // 2. Check for sign
    if (s[i] == '+' || s[i] == '-') {
        sign = (s[i] == '-') ? -1 : 1;
        i++;
    }

    // 3. Conversion and stopping at non-digits
    while (s[i] >= '0' && s[i] <= '9') {
        res = res * 10 + (s[i] - '0');

        // 4. Rounding (Overflow check)
        if (res * sign >= INT_MAX) return INT_MAX;
        if (res * sign <= INT_MIN) return INT_MIN;
        i++;
    }

    return (int)(res * sign);
}

int main() {

  char s[] = "-042-++";//{'1','2','2'};

  printf("%d\n", myAtoi(s));
  /* printf("Sizeof : %ld\n", strlen(s)); */
  return 0;
}
