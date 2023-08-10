#include "raylib.h"
#include "raymath.h"

typedef struct{
    f32 data[128*128];
    float label;
} Sample;

float Sigmoid(float z) { return 1/(1+(expf(-z)));}
float Relu(float z) { return (z>0) ? z : 0.0f;}
float dSigmoid(float z){ return z*(1-z);}
float dRelu(float z){ return (z>0) ? 1 : 0.0f;}
f64 BinaryCrossEntropy(f64 y, f64 a) { return -(y*log(a)+(1-y)*(1-y)*log(1-a)); }

int main(void) /**************MAIN***************/
{
    int screenWidth = 320;
    int screenHeight = 240;
    InitWindow(screenWidth, screenHeight, "emi");
    SetTargetFPS(0);
    SetWindowState(FLAG_WINDOW_TOPMOST);
    
    FilePathList catFiles = LoadDirectoryFilesEx("data/training/cats", ".jpg", true);
    FilePathList dogFiles = LoadDirectoryFilesEx("data/training/dogs", ".jpg", true);
    int numSamples = dogFiles.count + catFiles.count;
    Texture* textures = MemAlloc(numSamples * sizeof(Texture));
    Sample*   samples = MemAlloc(numSamples * sizeof(Sample));
    
    for(u32 i = 0; i < catFiles.count; i++)
    {
        Image catImage = LoadImage(catFiles.paths[i]);
        ImageResize(&catImage, 128, 128);
        ImageFormat(&catImage, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);
        textures[i] = LoadTextureFromImage(catImage);
        
        u8*  source = catImage.data;
        f32* target = samples[i].data;
        for(int pixel = 0; pixel < 128*128; pixel++) 
            *target = (float)(*source++/255.0f);
        samples[i].label = 0;
        
        UnloadImage(catImage);
    }
    for(u32 i = 0; i < dogFiles.count; i++)
    {
        Image dogImage = LoadImage(dogFiles.paths[i]);
        ImageResize(&dogImage, 128, 128);
        ImageFormat(&dogImage, PIXELFORMAT_UNCOMPRESSED_GRAYSCALE);
        textures[catFiles.count+i] = LoadTextureFromImage(dogImage);
        
        u8*  source = dogImage.data;
        f32* target = samples[catFiles.count+i].data;
        for(int pixel = 0; pixel < 128*128; pixel++) 
            *target = (float)(*source++/255.0f);
        samples[catFiles.count+i].label = 1;
        
        UnloadImage(dogImage);
    }
    
    f32* w1 = MemAlloc( 7 * 128*128 * sizeof(f32));
    for(int i = 0; i < 7 * 128*128; i++) w1[i] = GetRandomValue(-1000, 1000)/1000.0f;
    f32* b1 = MemAlloc(7 * sizeof(f32));
    for(int i = 0; i < 7; i++) b1[i] = 0;
    f32* z1 = MemAlloc(numSamples * 7 * sizeof(f32));
    f32* a1 = MemAlloc(numSamples * 7 * sizeof(f32));
    
    f32* w2 = MemAlloc(1 * 7 * sizeof(f32));
    for(int i = 0; i < 1 * 7; i++) w2[i] = GetRandomValue(-1000, 1000)/10000.0f;
    f32* b2 = MemAlloc(1 * sizeof(f32));
    for(int i = 0; i < 1; i++) b2[i] = 0;
    f32* z2 = MemAlloc(numSamples * 1 * sizeof(f32));
    f32* a2 = MemAlloc(numSamples * 1 * sizeof(f32));
    
    f32* dz2 = MemAlloc(numSamples * sizeof(f32));
    
    f32* dw2 = MemAlloc(1 * 7 * sizeof(f32));
    f32* db2 = MemAlloc(1 * sizeof(f32));
    
    f32* dz1 = MemAlloc(numSamples * sizeof(f32));
    
    f32* dw1 = MemAlloc(7 * 128*128 * sizeof(f32));
    f32* db1 = MemAlloc(7 * sizeof(f32));
    
    int selected = 0;
    float learningRate = 0.002f;
    
    while(!WindowShouldClose())
    {
        // PROPAGATION
        int units = 7;
        int inputs = 128*128;
        for(int i = 0; i < numSamples; i++) // per sample
        {
            for(int unit = 0; unit < units; unit++) // per unit
            {
                f32 sum = 0;
                for (int k = 0; k < inputs; k++) // per input
                    sum += w1[inputs*unit+k]*samples[i].data[k];
                z1[i*units+unit] = sum + b1[unit];
                a1[i*units+unit] = Relu(z1[i*units+unit]);
            }
        }
        
        units = 1;
        inputs = 7;
        for(int i = 0; i < numSamples; i++) // per sample
        {
            for(int unit = 0; unit < units; unit++) // per unit
            {
                f32 sum = 0;
                for (int k = 0; k < inputs; k++) // per input
                    sum += w2[inputs*unit+k]*a1[i*inputs+k]; // check activation
                z2[i*units+unit] = sum + b2[unit];
                a2[i*units+unit] = Sigmoid(z2[i*units+unit]);
            }
        }
        // COST
        f64 cost = 0.0;
        for(int i = 0; i < numSamples; i++)
        {
            f64 error = BinaryCrossEntropy(samples[i].label, a2[i]);
            cost += error;
        }
        cost /= numSamples;
        if (cost != cost) cost = 1000.0f; // if is nan
        
        // DW2
        for(int i = 0; i < numSamples; i++) dz2[i] = a2[i] - samples[i].label;
        
        units = 1;
        inputs = 7;
        for(int i = 0; i < numSamples; i++)
        {
            for(int unit = 0; unit < units; unit++)
            {
                for(int input = 0; input < inputs; input++)
                {
                    dw2[unit*inputs+input] +=  (1.0f/numSamples)*a1[numSamples*inputs+input]* dz2[i];
                }
            }
            
        }
        for(int i = 0; i < numSamples; i++) 
        {
            for(int j = 0; j < units; j++)
            {
                db2[j] += (1.0f/numSamples)*dz2[i];
            }
        }
        
        // DW1
        units = 7;
        inputs = 128*128;
        for(int i = 0; i < numSamples; i++)
        {
            for(int unit = 0; unit < units; unit++)
            {
                dz1[i] = w2[unit]*dz2[i] * dRelu(z1[i*units+unit]);
            }
        }
        
        for(int i = 0; i < numSamples; i++)
        {
            for(int unit = 0; unit < units; unit++)
            {
                for(int input = 0; input < inputs; input++)
                {
                    dw1[unit*inputs+input] +=  (1.0f/numSamples)*samples[i].data[input]* dz1[i];
                }
            }
            
        }
        for(int i = 0; i < numSamples; i++) 
        {
            for(int j = 0; j < units; j++)
            {
                db1[j] += (1.0f/numSamples)*dz1[i];
            }
        }
        
#if 0    
        // INFERENCE
        f32 activation;
        units = 7;
        inputs = 128*128;
        for(int unit = 0; unit < units; unit++) // per unit
        {
            f32 sum = 0;
            for (int k = 0; k < inputs; k++) // per input
                sum += w1[inputs*unit+k]*samples[i].data[k];
            activation = Relu(sum + b1[unit]);
        }
        
        units = 1;
        inputs = 7;
        
        for(int unit = 0; unit < units; unit++) // per unit
        {
            f32 sum = 0;
            for (int k = 0; k < inputs; k++) // per input
                sum += w2[inputs*unit+k]*a1[i*inputs+k]; // check activation
            z2[i*units+unit] = sum + b2[unit];
            a2[i*units+unit] = Relu(z2[i*units+unit]);
        }
#endif
        
        // UPDATE
        for(int unit = 0; unit < 1; unit++) for(int i = 0; i < 7; i++) w2[i] -= dw2[i] * learningRate;
        for(int unit = 0; unit < 1; unit++) b2[unit] -= db2[unit] * learningRate;
        for(int unit = 0; unit < 7; unit++)
        {
            for(int input = 0; input < 128*128; input++)
            {
                w1[unit*128*128+input] -= dw1[unit*128*128+input] * learningRate;
            }
        }
        for(int unit = 0; unit < 7; unit++) b1[unit] -= db1[unit] * learningRate;
        
        if(IsKeyPressed(KEY_P)) SetTargetFPS(1);
        if(IsKeyPressed(KEY_O)) SetTargetFPS(0);
        if(IsKeyDown(KEY_UP)) learningRate *= 2;
        if(IsKeyDown(KEY_DOWN)) learningRate /= 2;
        if(!IsWindowFocused()) SetWindowOpacity(0.3);
        else SetWindowOpacity(1.0);
        
        BeginDrawing();
        ClearBackground(DARKGRAY);
        DrawText(TextFormat("deltaTime: %f", GetFrameTime()), (int)(screenWidth*0.8), 0, 10, WHITE);
        DrawText(TextFormat("Cost: %f", cost), 0, 0, 40, (cost <= 1.0 ) ? GREEN : RED);
        DrawText(TextFormat("learning rate: %.8f", learningRate), 0, 40, 20,WHITE);
        DrawText(TextFormat("db2: %f", *db2), 0, 60, 20,WHITE);
        
        DrawTexture(textures[selected], (int)(screenWidth*0.41), (int)(screenHeight*0.4), WHITE);
        selected = GetRandomValue(0, numSamples-1);
        
        EndDrawing();
    }
    CloseWindow();
}
