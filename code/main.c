#include "raylib.h"
#include "raymath.h"

f32 Sigmoid(f32 z) {return 1/(1+expf(-z));}
f64 BinaryCrossEntropy(f32 y, f32 a) {return -(y*log(a)+(1-y)*log(1-a));}

int main(void) 
{
    int screenWidth = 320;
    int screenHeight = 240;
    InitWindow(screenWidth, screenHeight, "emi");
    SetTargetFPS(0);
    SetWindowState(FLAG_WINDOW_TOPMOST);
    
    FilePathList catFiles = LoadDirectoryFilesEx("data/training/cats", ".jpg", true);
    FilePathList dogFiles = LoadDirectoryFilesEx("data/training/dogs", ".jpg", true);
    int totalCount = catFiles.count + dogFiles.count;
    Image* images = MemAlloc(totalCount * sizeof(Image));
    Texture* textures = MemAlloc(totalCount * sizeof(Texture));
    f32* y = MemAlloc(totalCount * sizeof(f32));
    for(u32 i = 0; i < catFiles.count; i++)
    {
        Image image = LoadImage(catFiles.paths[i]);
        ImageResize(&image, 128, 128);
        textures[i] = LoadTextureFromImage(image);
        ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R32);
        y[i] = 0.0f;
        
        images[i] = image;
    }
    UnloadDirectoryFiles(catFiles);
    for(u32 i = 0; i < dogFiles.count; i++)
    {
        Image image = LoadImage(dogFiles.paths[i]);
        ImageResize(&image, 128, 128);
        textures[catFiles.count+i] = LoadTextureFromImage(image);
        ImageFormat(&image, PIXELFORMAT_UNCOMPRESSED_R32);
        y[catFiles.count+i] = 1.0f;
        images[catFiles.count+i] = image;
    }
    UnloadDirectoryFiles(dogFiles);
    
    f32* w = MemAlloc(1 * 128*128 * sizeof(f32));
    f32* b = MemAlloc(1 * sizeof(f32));
    f32* dw= MemAlloc(1 * 128*128 * sizeof(f32));
    f32* db= MemAlloc(1 * sizeof(f32));
#if 1
    u32 bytesRead = 0;
    f32* weightsToRead = (float *)LoadFileData("weights.emi", &bytesRead);
    f32* weight = weightsToRead;
    for(int i = 0; i < 128*128; i++) w[i] = *weight++;
    for(int i = 0; i < 1; i++) b[i] = *weight++;
    for(int i = 0; i < 128*128; i++) dw[i] = *weight++;
    for(int i = 0; i < 1; i++) db[i] = *weight++;
#else
    for(int i = 0; i < 128*128; i++) w[i] = GetRandomValue(-1000, 1000)/1000000.0f;
#endif
    f32* z = MemAlloc(totalCount * 1 * sizeof(f32));
    f32* a = MemAlloc(totalCount * 1 * sizeof(f32));
    f32* dz= MemAlloc(totalCount * sizeof(f32));
    f32* sdw = MemAlloc(1 * 128*128 * sizeof(f32));
    f32* sdb = MemAlloc(1 * sizeof(f32));
    f32* savedWeights = MemAlloc(1*128*128*sizeof(f32)*2 + 1*sizeof(f32)*2);
    
    int selected = 0;
    
    f32 learningRate = 1.00;
    f32 beta1 = 0.9f;
    f32 beta2 = 0.999f;
    int totalCounter = 0;
    int rightCounter = 0;
    f32 precision;
    f32 precisionOld = 0;
    f64 deltaTime = GetTime();
    while(!WindowShouldClose())
    {
        f32* data = images[selected].data;
        f32 sum = 0;
        for(int input = 0; input < 128*128; input++) sum += w[input]*data[input];
        z[selected] = sum + b[0];
        a[selected] = Sigmoid(z[selected]);
        
        f64 cost = BinaryCrossEntropy(y[selected], a[selected]);
        
        dz[selected] = a[selected] - y[selected];
        for(int input = 0; input < 128*128; input++) dw[input] += data[input]*dz[selected];
        db[0] += dz[selected];
        
        for(int input = 0; input < 128*128; input++) sdw[input] = beta2*sdw[input] + (1-beta2)*dw[input]*dw[input];
        sdb[0] = beta2*sdb[0] + (1-beta2)*db[0]*db[0];
        for(int input = 0; input < 128*128; input++) w[input] -= learningRate*dw[input]/sqrtf(sdw[input]);
        b[0] -= learningRate*db[0]/sqrtf(sdb[0]);
        
        bool right = (a[selected] >= 0.5 && y[selected] == 1.0) ||  (a[selected] < 0.5 && y[selected] == 0.0);
        if (right) rightCounter += 1;
        totalCounter += 1;
        precision = (f32)(rightCounter)*100 / totalCounter;
        
        if (precision >= precisionOld+0.1 && (GetTime()>=35))
        {
            precisionOld = precision;
            deltaTime = GetTime() - deltaTime;
        }
        
        if(IsKeyPressed('P')) SetTargetFPS(1);
        if(IsKeyPressed('O')) SetTargetFPS(0);
        if(IsKeyPressed(KEY_UP)) learningRate *= 2;
        if(IsKeyPressed(KEY_DOWN)) learningRate /= 2;
        if(!IsWindowFocused()) SetWindowOpacity(1);
        else SetWindowOpacity(1.0);
        if(IsMouseButtonPressed(0) && 
           CheckCollisionPointCircle((Vector2){(f32)(GetMouseX()), (f32)(GetMouseY())}, (Vector2){screenWidth*0.88f, screenHeight*0.5f}, 30))
        {
            f32* to = savedWeights;
            f32* from = w;
            for(int i = 0; i < 1*128*128; i++) *to++ = *from++;
            from = b;
            for(int i = 0; i < 1; i++) *to++ = *from++;
            from = dw;
            for(int i = 0; i < 1*128*128; i++) *to++ = *from++;
            from = db;
            for(int i = 0; i < 1; i++) *to++ = *from++;
            SaveFileData("weights.emi", savedWeights, 1*128*128*sizeof(f32)*2 + 1*sizeof(f32)*2);
        }
        
        BeginDrawing();
        ClearBackground(DARKGRAY);
        //DrawFPS((int)(screenWidth*0.80), 0);
        SetWindowTitle(TextFormat("emi (FPS: %i)", GetFPS()));
        DrawTexture(textures[selected], (int)(screenWidth*0.32), (int)(screenHeight*0.23), WHITE);
        DrawText(TextFormat("Precision: %.3f%%", precision), (int)(screenWidth*0.1), (int)(screenHeight*0.05), 30, WHITE);
        DrawText(TextFormat("dt: %f", deltaTime), (int)(screenWidth*0.6), (int)(screenHeight*0.17), 10, GREEN);
        DrawText(TextFormat("oldPrecision: %.3f%%", precisionOld), (int)(screenWidth*0.1), (int)(screenHeight*0.17), 10, WHITE);
        DrawText(TextFormat("is: %s",y[selected]<0.5?"cat":"dog"), (int)(screenWidth*0.43), (int)(screenHeight*0.80), 20, WHITE);
        DrawText(TextFormat("predicted: %s", a[selected]<0.5?"cat":"dog"), (int)(screenWidth*0.30), (int)(screenHeight*0.88), 20, WHITE);
        DrawText(TextFormat("lr: %f",learningRate), (int)(screenWidth*0.01), (int)(screenHeight*0.46), 10, WHITE);
        DrawText(TextFormat("dz: %.2f",dz[selected]), (int)(screenWidth*0.01), (int)(screenHeight*0.50), 10, WHITE);
        DrawText(TextFormat("b: %.2f",b[0]), (int)(screenWidth*0.01), (int)(screenHeight*0.54), 10, WHITE);
        DrawText(TextFormat("db: %.2f",db[0]), (int)(screenWidth*0.01), (int)(screenHeight*0.58), 10, WHITE);
        DrawCircle((int)(screenWidth*0.88), (int)(screenHeight*0.50), 30, right ? GREEN : RED);
        EndDrawing();
        selected = GetRandomValue(0, totalCount-1);
    }
    CloseWindow();
}