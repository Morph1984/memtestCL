/*
 * memtestCL_core.h
 * Public API for core memory test functions for MemtestCL
 * Includes functional and OO interfaces to GPU test functions.
 *
 * Author: Imran Haque, 2010
 * Copyright 2010, Stanford University
 *
 * This file is licensed under the terms of the LGPL. Please see
 * the COPYING file in the accompanying source distribution for
 * full license terms.
 *
 */
#ifndef _MEMTESTCL_CORE_H_
#define _MEMTESTCL_CORE_H_

#include <stdio.h>
#include <iostream>
#include <list>
#include "types.h"

using namespace std;

#if defined (WINDOWS) || defined (WINNV)
    #include <windows.h>
    inline u32 getTimeMilliseconds(void) {
        return GetTickCount();
    }
    #include <windows.h>
	#define SLEEPMS(x) Sleep(x)
#elif defined (LINUX) || defined (OSX)
    #include <sys/time.h>
    inline u32 getTimeMilliseconds(void) {
        struct timeval tv;
        gettimeofday(&tv,NULL);
        return tv.tv_sec*1000 + tv.tv_usec/1000;
    }
    #include <unistd.h>
    #define SLEEPMS(x) usleep(x*1000)
#else
    #error Must #define LINUX, WINDOWS, WINNV, or OSX
#endif

#if defined (__APPLE__) || defined(MACOSX) || defined(OSX)
   #include <OpenCL/opencl.h>
#else
   #include <CL/opencl.h>
#endif

cl_int softwaitForEvents(cl_uint num_events,const cl_event* event_list,cl_command_queue const* pcq=NULL,unsigned sleeplength=1,unsigned limit=15000);


const char* descriptionOfError (cl_int err);

// Low-level OO interface to MemtestCL functions
class memtestFunctions { //{{{
protected:
    cl_context ctx;
    cl_device_id dev;
    cl_command_queue cq;
    cl_program code;
    static const s32 n_kernels = 12;
    cl_kernel kernels[n_kernels];
    cl_kernel &k_write_constant, &k_verify_constant;
    cl_kernel &k_logic,&k_logic_shared;
    cl_kernel &k_write_paired_constants,&k_verify_paired_constants;
    cl_kernel &k_write_w32,&k_verify_w32;
    cl_kernel &k_write_random,&k_verify_random;
    cl_kernel &k_write_mod,&k_verify_mod;
    cl_int setKernelArgs(cl_kernel& kernel,const s32 n_args,const size_t* sizes,const void** args) const;
public:
    memtestFunctions(cl_context context,cl_device_id device,cl_command_queue q);
    ~memtestFunctions();
    u32 max_workgroup_size() const;
    cl_event writeConstant(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 constant,cl_int& status) const;
    cl_event writePairedConstants(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 constant1,const u32 constant2,cl_int& status) const;
    cl_event writeWalking32Bit(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const bool ones,const u32 shift,cl_int& status) const;
    cl_event writeRandomBlocks(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 seed,cl_int& status) const;
    cl_event writePairedModulo(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 shift,const u32 pattern1, const u32 pattern2, const u32 modulus,const u32 iters,cl_int& status) const;
    cl_event shortLCG0(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 repeats,const u32 period,cl_int& status) const;
    cl_event shortLCG0Shmem(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 repeats,const u32 period,cl_int& status) const;
    u32 verifyConstant(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 constant,cl_mem blockErrorCount,u32* error_counts,cl_int& status) const;
    u32 verifyPairedConstants(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 constant1,const u32 constant2,cl_mem blockErrorCount,u32* error_counts,cl_int& status) const;
    u32 verifyWalking32Bit(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const bool ones,const u32 shift,cl_mem blockErrorCount,u32* error_counts,cl_int& status) const;
    u32 verifyRandomBlocks(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 seed,cl_mem blockErrorCount,u32* error_counts,cl_int& status) const;
    u32 verifyPairedModulo(const u32 nBlocks,const u32 nThreads,cl_mem base,u32 N,const u32 shift,const u32 pattern1,const u32 modulus,cl_mem blockErrorCount,u32* error_counts,cl_int& status) const;

}; //}}}

// OO interface to MemtestCL functions
class memtestState { //{{{
    friend class memtestMultiTester;
protected:
    cl_context ctx;
    cl_device_id dev;
    cl_command_queue cq;
    memtestFunctions memtest;
	u32 nBlocks;
	u32 nThreads;
    u32 loopFactor;
    u32 loopIters;
	u32 megsToTest;
    s32 lcgPeriod;
	cl_mem devTestMem;
	cl_mem devTempMem;
	bool allocated;
	u32* hostTempMem;
	bool writeConstant(const u32 constant) const;
	bool verifyConstant(u32& errorCount,const u32 constant) const;
	bool gpuMovingInversionsPattern(u32& errorCount,const u32 pattern) const;
public:
    u32 initTime;
	memtestState(cl_context context, cl_device_id device);
    ~memtestState();

	u32 allocate(u32 mbToTest);
	void deallocate();
	bool isAllocated() const {return allocated;}
	u32 size() const {return megsToTest;}
    void setLCGPeriod(s32 period) {lcgPeriod = period;}
    s32 getLCGPeriod() const {return lcgPeriod;}
    u32 max_bandwidth_size() const {return megsToTest/2;}
    u32 workgroup_size() const {return nThreads;}

    bool gpuMemoryBandwidth(double& bandwidth,u32 mbToTest,u32 iters=5);
	bool gpuShortLCG0(u32& errorCount,const u32 repeats) const;
	bool gpuShortLCG0Shmem(u32& errorCount,const u32 repeats) const;
	bool gpuMovingInversionsOnesZeros(u32& errorCount) const;
	bool gpuWalking8BitM86(u32& errorCount,const u32 shift) const;
	bool gpuWalking8Bit(u32& errorCount,const bool ones,const u32 shift) const;
	bool gpuMovingInversionsRandom(u32& errorCount) const;
	bool gpuWalking32Bit(u32& errorCount,const bool ones,const u32 shift) const;
	bool gpuRandomBlocks(u32& errorCount,const u32 seed) const;
	bool gpuModuloX(u32& errorCount,const u32 shift,const u32 pattern,const u32 modulus,const u32 overwriteIters) const;
}; //}}}

// Simple wrapper class around memtestState to allow multiple test regions
class memtestMultiTester {
    protected:
    list<memtestState*> testers;
    cl_context ctx;
    cl_device_id dev;
    u32 lcg_period;
    bool ctx_retained;
    u32 allocation_unit;
    memtestMultiTester(cl_device_id device) : dev(device), lcg_period(1024), ctx_retained(false), initTime(0)
    {
        cl_ulong maxalloc;
        clGetDeviceInfo(dev,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&maxalloc,NULL);
        // in MiB
        allocation_unit = (u32)(maxalloc/1048576);
    }
    public:
    u32 initTime;
	memtestMultiTester(cl_context context, cl_device_id device) : ctx(context), dev(device), lcg_period(1024), ctx_retained(true), initTime(0)
    { //{{{
        clRetainContext(ctx);
        cl_ulong maxalloc;
        clGetDeviceInfo(dev,CL_DEVICE_MAX_MEM_ALLOC_SIZE,sizeof(cl_ulong),&maxalloc,NULL);
        // in MiB
        allocation_unit = (u32)(maxalloc/1048576);
    }; //}}}
    virtual ~memtestMultiTester() {
        deallocate();
        if (ctx_retained) clReleaseContext(ctx);
    }

    s32 getLCGPeriod() const {return lcg_period;}
    u32 get_allocation_unit() const {return allocation_unit;}
	bool isAllocated() const {return testers.size()>0;}
	u32 size() const {
        u32 totalsize = 0;
        for (list<memtestState*>::const_iterator i = testers.begin(); i != testers.end(); ++i) {
            totalsize += (*i)->size();
        }
        return totalsize;
    }
    u32 max_bandwidth_size() const {
            if (!isAllocated()) return 0;
            return testers.front()->size()/2;
    }
    u32 workgroup_size() const {
            if (!isAllocated()) return 0;
            return testers.front()->workgroup_size();
    }
    void setLCGPeriod(s32 period) {
        lcg_period = period;
        for (list<memtestState*>::iterator i = testers.begin(); i != testers.end(); ++i) {
            (*i)->setLCGPeriod(period);
        }
    }

	virtual u32 allocate(u32 mbToTest);
	virtual void deallocate();
    bool gpuMemoryBandwidth(double& bandwidth,u32 mbToTest,u32 iters=5);
	bool gpuShortLCG0(u32& errorCount,const u32 repeats) const;
	bool gpuShortLCG0Shmem(u32& errorCount,const u32 repeats) const;
	bool gpuMovingInversionsOnesZeros(u32& errorCount) const;
	bool gpuWalking8BitM86(u32& errorCount,const u32 shift) const;
	bool gpuWalking8Bit(u32& errorCount,const bool ones,const u32 shift) const;
	bool gpuMovingInversionsRandom(u32& errorCount) const;
	bool gpuWalking32Bit(u32& errorCount,const bool ones,const u32 shift) const;
	bool gpuRandomBlocks(u32& errorCount,const u32 seed) const;
	bool gpuModuloX(u32& errorCount,const u32 shift,const u32 pattern,const u32 modulus,const u32 overwriteIters) const;
}; //}}}

class memtestMultiContextTester : public memtestMultiTester {
    protected:
        cl_platform_id plat;
    public:
        memtestMultiContextTester(cl_platform_id platform,cl_device_id device) : memtestMultiTester(device), plat(platform) {}
        virtual ~memtestMultiContextTester() {};
        virtual u32 allocate(u32 mbToTest);
};


#endif
