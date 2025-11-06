# NVFLAGS := -std=c++11 -O3 
# TARGET := hw4

# .PHONY: all
# all: $(TARGET)

# # $(TARGET): sha256.o hw4.cu
# # 	nvcc $(NVFLAGS) -o hw4 hw4.cu sha256.o

# # sha256.o: sha256.cu
# # 	nvcc $(NVFLAGS) -c sha256.cu

# # $(TARGET): hw4.cu sha256.cu sha256.h
# # 	nvcc $(NVFLAGS) -o $(TARGET) hw4.cu sha256.cu

# $(TARGET): hw4.cu sha256.cu sha256.h
# 	nvcc $(NVFLAGS) -o $(TARGET) hw4.cu sha256.cu

# clean:
# 	rm -rf hw4 *.o




NVFLAGS := -std=c++11 -O3 
TARGET := hw4
OBJS := hw4.o sha256.o

.PHONY: all
all: $(TARGET)

# 連結步驟：將 .o 檔案連結成可執行檔
$(TARGET): $(OBJS)
	nvcc $(NVFLAGS) -o $(TARGET) $(OBJS)

# 編譯步驟：將 .cu 編譯成 .o，並使用 -dc 旗標
hw4.o: hw4.cu sha256.h
	nvcc $(NVFLAGS) -dc -o hw4.o hw4.cu

sha256.o: sha256.cu sha256.h
	nvcc $(NVFLAGS) -dc -o sha256.o sha256.cu

clean:
	rm -rf $(TARGET) *.o