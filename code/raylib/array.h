#ifndef ARRAY_H
#define ARRAY_H

#include <stdlib.h>

typedef struct Array
{
    int count;
    int capacity;
    float* numbers;
} Array;

void WriteArray(Array* array, float number);
void InitArray(Array* array);

typedef struct ArrayV
{
    int capacity;
    int count;
    Vector3* vectors;
} ArrayV;
void WriteArrayV(ArrayV* array, Vector3 vector);
void InitArrayV(ArrayV* array);

typedef struct ArrayA
{
    int count;
    int capacity;
    Array* arrays;
} ArrayA;

void WriteArrayA(ArrayA* arrayA, Array A);
void InitArrayA(ArrayA* arrayA);


#endif //ARRAY_H

void WriteArray(Array* array, float number)
{
    if (array->capacity < array->count + 1) // if not enough capacity
    {
        array->capacity = (array->capacity < 8) ? 8: array->capacity*2; // update capacity
        array->numbers = (float *)realloc(array->numbers, sizeof(float)*array->capacity);//reallocate
    }
    
    array->numbers[array->count++] = number;
}

void InitArray(Array* array)
{
    free(array->numbers);
    *array = (Array){0};
}

void WriteArrayV(ArrayV* array, Vector3 vector)
{
    if (array->capacity < array->count + 1) // if not enough capacity
    {
        array->capacity = (array->capacity < 8) ? 8: array->capacity*2; // update capacity
        array->vectors = (Vector3 *)realloc(array->vectors, sizeof(Vector3)*array->capacity);//reallocate
    }
    array->vectors[array->count++] = vector;
}

void InitArrayV(ArrayV* array)
{
    free(array->vectors);
    *array = (ArrayV){0};
}

void WriteArrayA(ArrayA* arrayA, Array array)
{
    if (arrayA->capacity < arrayA->count + 1) // if not enough capacity
    {
        arrayA->capacity = (arrayA->capacity < 8) ? 8: arrayA->capacity*2; // update capacity
        arrayA->arrays = (Array *)realloc(arrayA->arrays, sizeof(Array)*arrayA->capacity);//reallocate
    }
    
    arrayA->arrays[arrayA->count++] = array;
}

void InitArrayA(ArrayA* arrayA)
{
    free(arrayA->arrays);
    *arrayA = (ArrayA){0};
}
