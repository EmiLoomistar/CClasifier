#include "raylib.h"
#include "raymath.h"

typedef struct{
    f32 data[128*128];
    f32 label;
} Sample;

f32 Sigmoid(f32 z) { return 1/(1+expf(-z));}
f32 Relu(f32 z) { return (z>0) ? z : 0.0f;}
f64 BinaryCrossEntropy(f64 y, f64 a) { return -( y*log(a)+(1.0-y)*(1-y)*log(1.0-a) ); }

int main(void) /**************MAIN***************/
{
    int screenWidth = 640;
    int screenHeight = 480;
    InitWindow(screenWidth, screenHeight, "emi");
    SetTargetFPS(0);
    
    FilePathList catFiles = LoadDirectoryFilesEx("data/trainingMini/cats", ".jpg", true);
    FilePathList dogFiles = LoadDirectoryFilesEx("data/trainingMini/dogs", ".jpg", true);
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
            *target = (f32)(*source++)/255.0f;
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
            *target = (float)(*source++)/255.0f;
        samples[catFiles.count+i].label = 1;
        
        UnloadImage(dogImage);
    }
    
    f32* w = MemAlloc( 1 * 128*128 * sizeof(f32));
    for(int i = 0; i < 1 * 128*128; i++) w[i] = GetRandomValue(-1000, 1000)/1000000.0f;
    f32* b = MemAlloc(1 * sizeof(f32));
    for(int i = 0; i < 1; i++) b[i] = 0;
    f32* z = MemAlloc(numSamples * 1 * sizeof(f32));
    f32* a = MemAlloc(numSamples * 1 * sizeof(f32));
    
    f32* dz = MemAlloc(numSamples * sizeof(f32));
    
    f32* dw = MemAlloc(1 * 128*128 * sizeof(f32)); // same shape as w
    f32* db = MemAlloc(1 * sizeof(f32)); // same shape as b
    
    int selected = 0;
    f32 learningRate = 0.0000001f;
    
    while(!WindowShouldClose())
    {
        int units = 1;
        int inputs = 128*128;
        for(int i = 0; i < numSamples; i++) // per sample
        {
            for(int unit = 0; unit < units; unit++) // per unit
            {
                f32 sum = 0;
                for (int k = 0; k < inputs; k++) // per input
                    sum += w[inputs*unit+k]*samples[i].data[k];
                z[i*units+unit] = sum + b[unit];
                a[i*units+unit] = Sigmoid(z[i*units+unit]);
            }
        }
        
        f64 cost = 0.0;
        for(int i = 0; i < numSamples; i++)
        {
            f64 error = BinaryCrossEntropy((f64)(samples[i].label), (f64)(a[i]));
            cost += error;
        }
        cost /= numSamples;
        if (cost != cost) cost = 1000.0f; // if is nan
        
        for(int i = 0; i < numSamples; i++) 
        {
            for(int unit = 0; unit < 1; unit++)
            {
                dz[i] = a[i] - samples[i].label;
            }
        }
        
        units = 1;
        inputs = 128*128;
        for(int i = 0; i < numSamples; i++)
        {
            for(int unit = 0; unit < units; unit++)
            {
                for(int input = 0; input < inputs; input++)
                {
                    dw[unit+input] += (1.0f/numSamples)*samples[i].data[input] * dz[i];
                }
            }
            
        }
        for(int i = 0; i < numSamples; i++) 
        {
            for(int j = 0; j < units; j++)
            {
                db[j] += (1.0f/numSamples)*dz[i];
            }
        }
        
        for(int i = 0; i < inputs; i++)
        {
            w[i] -= dw[i] * learningRate;
        }
        
        for(int i = 0; i < units; i++)
        {
            b[i] -= db[i] * learningRate;
        }
        
        
        if(IsKeyPressed(KEY_P)) SetTargetFPS(4);
        if(IsKeyPressed(KEY_O)) SetTargetFPS(0);
        if(IsKeyPressed(KEY_UP)) learningRate *= 2;
        if(IsKeyPressed(KEY_DOWN)) learningRate /= 2;
        
        BeginDrawing();
        ClearBackground(DARKGRAY);
        DrawText(TextFormat("dt: %f", GetFrameTime()*1000), (int)(screenWidth*0.80f), 0, 20, WHITE);
        
        DrawText(TextFormat("Cost: %f", cost), 0, 0, 40, (cost <= 1.0 ) ? GREEN : RED);
        DrawText(TextFormat("learningRate: %.16f", learningRate), 0, 40, 10, WHITE);
        DrawText(TextFormat("b: %f", b[0]), 0, 50, 20, WHITE);
        DrawText(TextFormat("db: %f", db[0]), 0, 70, 20, WHITE);
        
        DrawTexture(textures[selected], (int)(screenWidth*0.41), (int)(screenHeight*0.4), WHITE);
        selected = GetRandomValue(0, numSamples-1);
        
        EndDrawing();
    }
    CloseWindow();
}
