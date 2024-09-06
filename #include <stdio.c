#include <stdio.h>
#include <math.h>

int main() {
    // Declare variables
    int num, i;
    int isPrime = 1; // Assume the number is prime initially

    // Get user input
    printf("Enter a number: ");
    scanf("%d", &num);

    // Check for divisibility
    for (i = 2; i <= sqrt(num); ++i) {
        if (num % i == 0) {
            isPrime = 0; // Number is divisible, so it's not prime
            break;
        }
    }

    // Display the result
    if (num <= 1) {
        printf("%d is not a prime number.\n", num);
    } else if (isPrime) {
        printf("%d is a prime number.\n", num);
    } else {
        printf("%d is not a prime number.\n", num);
    }

    return 0; // Exit successfully
}
