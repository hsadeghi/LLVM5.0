#define DEBUG_TYPE "indOpt"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/CFG.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/IR/ConstantRange.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Pass.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Transforms/Utils/UnifyFunctionExitNodes.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Analysis/LoopPass.h"
#include "llvm/Analysis/ScalarEvolutionExpander.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Type.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopIterator.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <stack>
#include <list>
#include <map>
#include <unordered_set>

using namespace std;
using namespace llvm;

namespace {
/*
 * each llvm value (instruction or basicblock) could be assigned by a type which indicates some level of optimization for corresponding value. for the purpose of this pass
 * (which is induction optimization),  4 following types is considered :
 *   __const : a compile time constant value
 *   __LI : compile time unknown, yet runtime time Loop Invariant value
 *   __Ind : an induction (or scalar) value, generally a value is considered scalar if exists a linear closed-formula of the main induction variable of the
 *   		loop (plus some __LI and __Const  values), main induction variable is a loop-caried dependency variable (a.k.a Loop Phi Node in SSA form)
 *   		which (i) its change (add, sub, multiply, shift ) is of type __LI or __const , and (ii) loop's exit condition is just dependent on variable (and perhaps some __LI values).
 *   		detecting and optimizing induction variables is of paramount importance since many other analysis and transformations are dependent on it (like unrolling,
 *   		vectorization, automatic parallelization, memory dependency analysis and ... ), during my PhD work I noticed that llvm scalar evolution and other induction recognitions passes fail to detect
 *   		varirty of induction variables since they do not take Loop invariant predicates into account, for instance consider following loop :
 *
 *   		for(i=c; i < n;){
 *   			// loop budy
 *   			if(a>b)  i+=c;
 *   			else      i+=d;
 *   		}
 *
 *   		which can be rewritten by compiler as:
 *
 *   		step = (a>b)? c : d;
 *   		for(i=c; i<n;  i+= step){
 *   			//loop budy
 *   		}
 *
 *			another important and common case which scalar evolution fails is to detect nested inductions (example IndNestedLoop.cpp in benchmarks).
 *
 *			Type __Cx stands for Complex type, when a value can not be proven to be of any ther type, the it is considered __Cx which is not optimizable (at least
 *			without any profiling or run-time info), the default type for an instruction or basicblock is __Cx until some facts can be proven in order to promote the type
 *
 */

enum class ValType {__Const, __LI, __Ind, __Cx}; // other potential types : Scalar_mem,  __IndirectScalar_mem , __PReduct,  __SReduct

struct Predicate{
	Value* pred;
	bool JumpCnd;
	Predicate(Value* p=nullptr, bool jc = false ): pred(p), JumpCnd(jc){};

};

/*
 * a tree like data structure for keeping and processing Loop Invariant information
 */
struct LIVal {
	Value* val; // val is computational operand for binary operators (sub/add) or a predicate value for Iselect (cmov) or conditional phi
	bool isNegetive; // to differentiate add/sub
	bool isPhi;
	LIVal* LeftDef;
	LIVal* RightDef;
	BasicBlock* DefBlock;
	LIVal(Value* v = nullptr, bool s=false, bool ispi = false, LIVal* l = nullptr, LIVal* r = nullptr, BasicBlock* bb=nullptr)
		: val(v), isNegetive(s), isPhi(ispi), LeftDef(l) , RightDef(r), DefBlock(bb){};
};

/*
 * instPlus extends llvm::Instruction class with value type
 */
class InstPlus {
	Instruction* inst;
	llvm::SmallDenseSet<PHINode* > Lphi;
	ValType ty;

public:

	LIVal* LiValue;
	InstPlus(Instruction* instP, ValType t = ValType::__Cx, LIVal* li = nullptr, InstPlus* r=nullptr, InstPlus* l=nullptr)
	: inst(instP), ty(t), LiValue(li) {};
	inline ValType getType() const { return ty;}
	inline void SetType(ValType  newT) {ty = newT;}
};

/*
 * BBplus extends LLVM::BasicBlock class with value type and some othe extra utilities regarding dominators and post dominators, also extra info regarding
 *  its dominator branch's condition (a.k.a BasicBlock's Predicate)
 */
class BBplus {
	BasicBlock* BB;
	ValType ty;
	BasicBlock* ImmDom;

public:
	Predicate PRD;
	BBplus(BasicBlock* bb, ValType t = ValType::__Cx ,  BasicBlock* pd= nullptr): BB(bb), ty(t), ImmDom(pd){};
	inline ValType getType() const {return ty;}
	inline void SetType(ValType  newT) {ty = newT;}
	inline BasicBlock* getIdom()const {return ImmDom;}
	~BBplus(){

	}
};

/*
 * LoopPlus: extension of llvm::Loop* class which does the main work, like analysing basicblocks, traversing use-def chains in the loop, keeping map of info regarding
 * BasicBlocks and instructions,  analaysing Loop PHi nodes to detect inductions, (re)constructing LoopInvar values in loop preheader,
 * optimizing inductions, ...
 */
class LoopPlus{
	  Loop* L;
	  DominatorTree* DT;
	  PostDominatorTree* PDT;
	  LoopInfo* LI;
	  std::vector<Instruction*> DeadInsts; // a list of dead instructions after this transform

	  llvm::DenseMap<Instruction*, InstPlus*> InstMap;
	  llvm::DenseMap<BasicBlock*, BBplus*>  BBmap;
	  std::vector<LIVal*> Livals;
	  llvm::DenseMap<Loop*, LoopPlus*>* ptrToMap;
	  PHINode* exitPhi; /* pointer to Loop's main induction which is a scalar variable (a.k.a loop counter) who determines loop's exit condition,  in the case
	  	  that such phi does not exist, this pass returns without any modification */

	  void printLIVs(const LIVal* Li)const; // just for debuging ...
	  void SetType( Instruction* inst,  ValType ty);
	  ValType getType( Instruction*) const;
	  ValType getType(const BasicBlock *) const;
	  BBplus* getBlockInfo(const BasicBlock*) const;
	  InstPlus* getInstInfo(const Instruction* ) const;
	  Value*  constructInductionStep(PHINode* Lphi); /* (re)construcs Loop Phi step in preh and returns the final value */
	  Value* cloneinPreh(Value* inst, Loop* L, BasicBlock* preh);
	  Value* RecursiveBuilder(IRBuilder<> &builder, BasicBlock* insertPoint, LIVal* LiV, llvm::Type* instType );
	  PHINode* backwardUseDef(Instruction* current, std::vector<Instruction*> &divergePoints, const  Loop* lp, DenseSet<Instruction*> &instSet,
			  BasicBlock* nextMegePoint);
	  inline BasicBlock* findIPostDom( BasicBlock* bb)const { return PDT->getNode(bb)->getIDom()->getBlock(); } // Immediate Post Dominator
	  inline BasicBlock* findIDom( BasicBlock* bb) const ;// Immediate  Dominator
	  inline Loop* getParentLoop(const Instruction* inst) const {return LI->getLoopFor(inst->getParent());};
	  BasicBlock* whereToInsret(Value* toMove);
	  LoopPlus* GetLoopPlusFor( Loop* L = nullptr) ;
	  void ReleaseMemory();

public:
	  LoopPlus(Loop* lp , DominatorTree* dt, PostDominatorTree* pdt , LoopInfo* li, std::vector<Instruction*> &deadI,
			  	  llvm::DenseMap<Loop*, LoopPlus*>* ptrMp = nullptr,PHINode* ephi = nullptr ) : L(lp), DT(dt), PDT(pdt) , LI(li) ,
				  DeadInsts(deadI),  ptrToMap(ptrMp), exitPhi(ephi) {};
	  bool PreProcess(); // initialzing instruction info map and detecting __LI and __const  instructions
	  bool BBAnalyzer(); // visiting blocks in reverse post order in order to determine __const and __LI blocks
	  inline Loop* getLoop(){return this->L;}
	  inline PHINode* getLoopExitPHi(){return this->exitPhi;}
	  bool isaLoopPhi( Instruction* inst) const;
	  bool AnalyseLoopPhisForInduction(); // detecs loop inductions
	  bool isInductionPhi(PHINode* loopPhi);
	  bool isBackEdgesInduction();
	  bool optimizeInductions();
	  Value* getInductionStart(PHINode* lphi);
	  Value* getLoopInductionEnd(Loop* L);
	  bool isInvariantofLoop( Value*, Loop* lp = nullptr)const;
	  ~LoopPlus(){ReleaseMemory();}

};

class IndOptimizer : public FunctionPass {

private:
  int numInnerLoops;
  int numRecPaths;

  LoopInfo *LI;
  DominatorTree* DT;
  PostDominatorTree*  PDT;
  std::vector<Loop*> RPOLoopVisi;

  std::vector<Instruction*> DeadInsts;
  llvm::DenseMap<Loop*, LoopPlus*> LoopInfoMap;

  void VisitLoopsRecursively(Loop* root);

  bool SCConLPhi(Loop *lp );
  void ReleaseMemory(); // release dynamically allocated memory


  public:
  static char ID;

  IndOptimizer() : FunctionPass(ID), numInnerLoops(0), numRecPaths(0) {
    initializeIndOptimizerPass(*PassRegistry::getPassRegistry());
  }

  virtual bool runOnFunction(Function &F);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;
};
}

char IndOptimizer::ID = 0;

INITIALIZE_PASS_BEGIN(IndOptimizer, "indOpt",
                      "Scalar Analysis and optimization", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)

INITIALIZE_PASS_DEPENDENCY(LoopInfoWrapperPass)

// INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(IndOptimizer, "indOpt",
                    "Scalar Analysis and optimization", false,
                    false)
FunctionPass *llvm::createIndOptimizerPass() { return new IndOptimizer(); }

void IndOptimizer::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.addRequired<LoopInfoWrapperPass>();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<PostDominatorTreeWrapperPass>();
}

bool IndOptimizer::runOnFunction(Function &F) {
//  DEBUG(dbgs() << " start working on function :\t " << F.getName() << "\n");

  LI = &getAnalysis<LoopInfoWrapperPass>().getLoopInfo();
  DT = &getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  PDT = &getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();

   LoopInfoMap.clear();
  DeadInsts.clear();
  RPOLoopVisi.clear();
  // find inner loops and insert them to the set
  for (auto T = LI->begin(), U = LI->end(); T != U; ++T) {
    Loop *Root = *T;
    VisitLoopsRecursively(Root);
    }

  auto changed  = false;
  for (auto I : RPOLoopVisi) {
    	Loop *Root = I;
    	DEBUG (dbgs() << "Loop name is " << Root->getHeader()->getName() << "\n");
    	auto LP = new LoopPlus(Root, DT, PDT, LI, DeadInsts, &LoopInfoMap);
    	LoopInfoMap[Root] = LP;
    	if(LP->PreProcess())
    		if(LP->BBAnalyzer())
    			if(LP->AnalyseLoopPhisForInduction())
    				if(LP->isBackEdgesInduction())
    					changed = LP->optimizeInductions();
    }

  ReleaseMemory();
  return changed;
}


void IndOptimizer::VisitLoopsRecursively(Loop* root){
	for(auto subL: root->getSubLoops()){
		VisitLoopsRecursively(subL);
	}
	RPOLoopVisi.push_back(root);
}

bool LoopPlus::AnalyseLoopPhisForInduction(){
	auto lp = this->L;
	bool hasIndPhi = false;
   	for(auto firstinst = lp->getHeader()->begin(); isa<PHINode>(firstinst); firstinst++){
    		DEBUG(dbgs()<<"analysing loopPhi : "<<*firstinst<<"\n");
    		if( isInductionPhi( dyn_cast<PHINode>(firstinst))) {
    			hasIndPhi = true;
    			//constructInductionStep(dyn_cast<PHINode>(firstinst));
    		}
    	}
   	return hasIndPhi;
}

Value* LoopPlus::cloneinPreh(Value* val, Loop* ll, BasicBlock* preh){
	auto lp = this->L;
	if(auto inst = dyn_cast<Instruction>(val))
		if(getParentLoop(inst) == lp){
			for(auto opr : inst->operand_values()){
				if(auto opinst = dyn_cast<Instruction>(opr)){
					if(lp->contains(opinst)){
						auto newOp =cloneinPreh(opinst, ll , preh);
						opinst->replaceAllUsesWith(newOp);
						DeadInsts.push_back(opinst);
					}
				}
			}
			Instruction *cloneVal = inst->clone();
			BasicBlock::iterator  it= preh->end();
			it--;
			preh->getInstList().insert( it , cloneVal);
			inst->replaceAllUsesWith(cloneVal);
			DeadInsts.push_back(inst);
			return cloneVal;
		}
	return val;
}

BasicBlock* LoopPlus::whereToInsret(Value *toMove){
	auto instToMove = dyn_cast<Instruction>(toMove);
	if(!instToMove) return nullptr;

	auto parentL = getParentLoop(instToMove);
	if(!parentL) return instToMove->getParent();

	BasicBlock* preh= nullptr;
	//Instruction* insertPt = nullptr;
	while(isInvariantofLoop(instToMove, parentL)){
		preh = parentL->getLoopPreheader();
		if(!preh) preh = parentL->getLoopPredecessor();
		assert(preh && "loop without predecessor !!");
		parentL = parentL->getParentLoop();
		if(!parentL) break;
	}
	if(!preh){
		DEBUG(dbgs()<<"can not find insert point for "<<*toMove<<"\n");
		return nullptr;
	}
	DEBUG(dbgs()<<" insert point for "<<*toMove<<"  is\t"<<preh->getName()<<"\n");
	return preh;
}



Value* LoopPlus::RecursiveBuilder(IRBuilder<> &builder, BasicBlock* insertPt, LIVal* LiV, llvm::Type* instType){
	builder.SetInsertPoint(insertPt->getTerminator());
	DEBUG(dbgs()<<"constructing LIval  : "<<LiV->DefBlock->getName()<<"\n");
	auto bb = LiV->DefBlock;
	if(DT->dominates(bb , insertPt)){
		if( LiV->val )return LiV->val;
		else return LiV->val =  ConstantInt::getNullValue(instType);
	}
	Loop* Lp = LI->getLoopFor(bb);
	if( !LiV->RightDef && !LiV->LeftDef ){
		LiV->DefBlock = insertPt;
		if(!LiV->val ) return ConstantInt::getNullValue(instType);
		else {
			if(!LiV->isNegetive){
				LiV->val = cloneinPreh(LiV->val, Lp , insertPt);
				return LiV->val;
			}else{
				Value* opR = cloneinPreh(LiV->val, Lp , insertPt);
				Value* opL = ConstantInt::getNullValue(instType);
				LiV->val = builder.CreateNSWSub(opL, opR, "sub.for.LI");
				LiV->isNegetive = false;
				 return LiV->val;
			}
		}
	}
	if(!LiV->RightDef && !LiV->isPhi){
		if( !LiV->val) return RecursiveBuilder(builder, insertPt, LiV->LeftDef, instType);
		auto LeftOp = RecursiveBuilder(builder, insertPt, LiV->LeftDef, instType);
		Value* result = nullptr;
		LiV->val = cloneinPreh(LiV->val , Lp , insertPt);
		if(LiV->isNegetive){
			DEBUG(dbgs()<<" creating sub .. \t ");
			result = builder.CreateNSWSub(LeftOp, LiV->val, "sub.for.LI");
		}else{
			DEBUG(dbgs()<<"creating add .. \t");
			result = builder.CreateNSWAdd(LeftOp, LiV->val, "add.for.LI");
		}
		LiV->DefBlock = insertPt;
		LiV->RightDef = nullptr;
		LiV->isNegetive = false;
		LiV->val = result;
		return result;
	}
	/*
	 * if both left and right exist then it is a phi node and can be reconst with cmov
	 */

	// TODO>> check if condition is safe to hoist for cmov an phi reconstruction
	auto lef= RecursiveBuilder(builder, insertPt, LiV->LeftDef, instType);
	Value* right = ConstantInt::getNullValue(instType);
	if(LiV->RightDef)	right = RecursiveBuilder(builder, insertPt, LiV->RightDef, instType);
	if(!LiV->isPhi){
			auto newCnd = cloneinPreh(LiV->val , Lp, insertPt);
			LiV->val = builder.CreateSelect(newCnd , lef , right, "iselClone");
			LiV->LeftDef = LiV->RightDef = nullptr;
			LiV->DefBlock = insertPt;
			return LiV->val;
	}
	auto bbinfo = getBlockInfo(bb);
	assert(bbinfo && "can not find bb info while constructing LI values !");
	auto condition = bbinfo->PRD.pred;
	if(!condition){
		DEBUG(dbgs()<<"no cond for left    "<<*lef<<"\t and right  :   "<<*right<<"\n");
		assert(condition && "invalid condition while constructing cmov!");
	}
	condition = cloneinPreh(condition , Lp, insertPt);
	DEBUG(dbgs()<<"creating select .. \t");
	LiV->val = builder.CreateSelect(condition, lef, right, "cmov.LI");
	LiV->LeftDef = LiV->RightDef = nullptr;
	LiV->isPhi = false; LiV->DefBlock = insertPt;
	return LiV->val;
}


Value* LoopPlus::constructInductionStep(PHINode* Lphi){
	if(!isInductionPhi(Lphi)) return nullptr;
	auto parent = getParentLoop(Lphi);
	auto lastDef = dyn_cast<Instruction>(Lphi->getIncomingValueForBlock(parent->getLoopLatch()));
	if(!lastDef) return nullptr;
	auto InstInf = getInstInfo(lastDef);
	auto preh = parent->getLoopPreheader();
	if(!preh) preh = parent->getLoopPredecessor();
	IRBuilder<> LiBuilder(preh->getTerminator());
	llvm::Type* ty = Lphi->getType();
	if(auto li = InstInf->LiValue)
		DEBUG(dbgs()<<li->DefBlock->getName());
	else { DEBUG(dbgs()<<"No LIinfofor" <<*lastDef<<" \n"); return nullptr;}
	return RecursiveBuilder(LiBuilder, preh, InstInf->LiValue, ty);
}

bool LoopPlus::isBackEdgesInduction( ){
	//if(!L->getExitBlock()) return false;
	if(this->exitPhi) return true;
	auto Latch = L->getLoopLatch();
	if(!Latch) return false;
	auto backEdge = dyn_cast<BranchInst>(Latch->getTerminator());
	while(backEdge && backEdge->isUnconditional()){
		if(Latch->getUniquePredecessor()){
			Latch = Latch->getUniquePredecessor();
			backEdge = dyn_cast<BranchInst>(Latch->getTerminator());
		}
		else return false;
	}
	if(!backEdge) return false;
	auto lastDef = dyn_cast<Instruction>(backEdge->getCondition());

	if(!L->contains(lastDef)) return false;
	if(!AnalyseLoopPhisForInduction()) return false;
	if(getType(lastDef) == ValType::__Ind ) {
		while(lastDef && !isaLoopPhi(lastDef)){
			for(auto defSite: lastDef->operand_values()){
				if(auto defInst = dyn_cast<Instruction>(defSite)){
					if(L->contains(defInst) && getType(defInst) == ValType::__Ind ) { lastDef = defInst; break;}
				}
			}
		}
		if( isaLoopPhi(lastDef) ){
			this->exitPhi = dyn_cast<PHINode>(lastDef);
			DEBUG(dbgs()<<"BackEdge is scalar \n");
			return true;
		}
	}
	return false;

}

Value* LoopPlus::getLoopInductionEnd(Loop* L){
	if(! this->exitPhi){
		if(!isBackEdgesInduction()) return nullptr;
	}
	auto Latch = L->getLoopLatch();
	if(!Latch) return nullptr;
	auto backEdge = dyn_cast<BranchInst>(Latch->getTerminator());
	while(backEdge && backEdge->isUnconditional()){
		if(Latch->getUniquePredecessor()){
			Latch = Latch->getUniquePredecessor();
			backEdge = dyn_cast<BranchInst>(Latch->getTerminator());
		}
		else return nullptr;
	}
	if(!backEdge) return nullptr;
	auto lastDef = dyn_cast<Instruction>(backEdge->getCondition());
	assert(lastDef && "no cmp for loop exit found !");
   for(auto cmpOpr :  lastDef->operand_values()){
	   if(!isa<Instruction>(cmpOpr)) return cmpOpr;
	   auto endVal = dyn_cast<Instruction>(cmpOpr);
	   if(!L->contains(endVal) || getType(endVal) <= ValType::__LI) return cmpOpr;
   }
   return nullptr;
}

Value* LoopPlus::getInductionStart(PHINode* Lphi){
	if(!isInductionPhi(Lphi)) return nullptr;
	auto parentL = getParentLoop(Lphi);
	auto preh = parentL->getLoopPreheader();
	if( !preh) preh = parentL->getLoopPredecessor();
	if(!preh) return nullptr;
	return Lphi->getIncomingValueForBlock(preh);
}

bool LoopPlus::optimizeInductions(){

	auto changed = false;
	if(!isBackEdgesInduction()) return false;
	auto mainInd = this->exitPhi;
	auto endInd = getLoopInductionEnd(L);
	auto startInd = getInductionStart(mainInd);
	if(! (mainInd && endInd && startInd)) return false;
	auto indStep = constructInductionStep(mainInd);
	assert(indStep && " can not compute main induction step !");
	DEBUG(dbgs()<<"\nmain ind \t"<<*mainInd<<"\n startVal \t"<<*startInd<<"\n end val \t"<<*endInd<<"\n step\t"<<*indStep<<"\n");

	// run time computation of #Loop_Iterations
	auto preh = L->getLoopPreheader();
	if(!preh) preh = L->getLoopPredecessor();
	llvm::Type*  tyInd = mainInd->getType();
	IRBuilder<> ScalarBuilder(preh->getTerminator());
	Value* ItRange = ScalarBuilder.CreateNSWSub(endInd , startInd, "iteratiobRange");
	Value* LoopIter0 = ScalarBuilder.CreateSDiv(ItRange,indStep , "LoopIter");
	Value* cndDiv = ScalarBuilder.CreateSRem(ItRange , indStep, "for.Isel");
	Value* zeroo = ConstantInt::getNullValue(tyInd);
	Value* cmp1 = ScalarBuilder.CreateICmpUGT(cndDiv,zeroo , "cmpForDiv" );
	Value* LoopIter1 = ScalarBuilder.CreateNSWAdd(LoopIter0 , ConstantInt::get(tyInd , 1));
	Value* LoopIter = ScalarBuilder.CreateSelect(cmp1 , LoopIter1 , LoopIter0 , "forRem");

	auto hdr = L->getHeader();
	auto latch = L->getLoopLatch();

	for(auto bbit = hdr->begin() ; isa<PHINode>(bbit) ; ++bbit){
		PHINode* Lphi = dyn_cast<PHINode>(bbit);
		if(Lphi->getType()->isVectorTy() ) continue; //  no support for vector types yet
		if(getType(Lphi) != ValType::__Ind) continue;
		Value* stp = constructInductionStep(Lphi);
		auto lastDef =  dyn_cast<Instruction>(Lphi->getIncomingValueForBlock(latch));

		auto insertPoint = dyn_cast<Instruction>(lastDef->getParent()->getFirstInsertionPt());
		ScalarBuilder.SetInsertPoint(insertPoint);
		auto newIncom = ScalarBuilder.CreateNSWAdd(Lphi , stp, "LastDef");
		lastDef->replaceAllUsesWith(newIncom);
		DeadInsts.push_back(lastDef);
		changed = true;

		ScalarBuilder.SetInsertPoint(preh->getTerminator());
		Value* startVal = this->getInductionStart(Lphi);
		auto exitbb = L->getExitBlock();
		if(!exitbb) continue;
		Value* totalchange = ScalarBuilder.CreateNSWMul(stp, LoopIter, "phiChange");
		Value* exitVal = ScalarBuilder.CreateNSWAdd(totalchange , startVal, "phiExitVal");

		for(auto usr : newIncom->users()){
			if(auto usrInst = dyn_cast<Instruction>(usr)){
				if(! L->contains(usrInst))
					usrInst->replaceUsesOfWith(newIncom , exitVal);
			}
		}

		for(auto usr : lastDef->users()){
			if(auto usrInst = dyn_cast<Instruction>(usr)){
				if(! L->contains(usrInst))
					usrInst->replaceUsesOfWith(newIncom , exitVal);
			}
		}


	}

	return changed;

}

bool LoopPlus::PreProcess(){
	auto lp = this->L;
	auto hdr = lp->getHeader();

	for(auto bbit = hdr->begin() ; isa<PHINode>(bbit) ; ++bbit){
		PHINode* Lphi = dyn_cast<PHINode>(bbit);
		if(Lphi->getType()->isVectorTy() ) return false; //  no support for vector types yet
	}
	LoopBlocksDFS dfsbb(lp);
	dfsbb.perform(LI);
	for(auto BBloop = dfsbb.beginRPO(); BBloop != dfsbb.endRPO(); ++BBloop){
		if(LI->getLoopFor(*BBloop) != this->L) continue;
		for(Instruction &inst : **BBloop){
			ValType TyInst = ValType::__Cx;
			if(isa<CallInst>(&inst) || isa<ReturnInst>(&inst)|| isa<FuncletPadInst>(&inst) ||
								isa<LandingPadInst>(&inst) || isa<FenceInst>(&inst) || isa<AtomicRMWInst>(&inst) || isa<AtomicCmpXchgInst>(&inst)
								|| isa<InsertElementInst>(&inst)) return false; // foe now  this pass covers kernel loops (with now call)
			if(const BranchInst* br=  dyn_cast<BranchInst>(&inst) ){
				if(br->isUnconditional()) TyInst = ValType::__Const;
				else if(isInvariantofLoop(&inst)) TyInst = ValType::__LI;
			}
			else if(isInvariantofLoop(&inst)) TyInst = ValType::__LI;

			InstPlus* newInst = new InstPlus(&inst, TyInst);
			InstMap[&inst] = newInst;
		}
	}

	// DEBUG: check if initial types are correct
	for(auto BBloop: lp->getBlocks()){
		for(Instruction &inst : *BBloop){
			if(getType(&inst) == ValType::__Const)
				DEBUG(dbgs()<<inst<<"\t ty :"<<"const"<<"\n");
			else if(getType(&inst) == ValType::__LI)
				DEBUG(dbgs()<<inst<<"\t ty :"<<"LI"<<"\n");

		}
	}
	return true;
}

bool LoopPlus::isaLoopPhi( Instruction* inst ) const {

	if(!inst) return false;

	if(PHINode* phi = dyn_cast<PHINode>(inst)){
			Loop* parentL = LI->getLoopFor(phi->getParent());
			if(parentL == nullptr) return false;
			if(parentL->getHeader() == phi->getParent()) return true;
	}
	return false;
}

BasicBlock* LoopPlus::findIDom(BasicBlock* bb)const {
	BasicBlock* idom =  DT->getNode(bb)->getIDom()->getBlock();
	while(auto pred = idom->getUniquePredecessor()){
		if(auto br = dyn_cast<BranchInst>(pred->getTerminator() ) )
				if (br->isUnconditional()) idom = pred;
				else break;
		else break;
	}
	return idom;
}

bool LoopPlus::BBAnalyzer() {
	auto lp = this->L;
	LoopBlocksDFS dfsbb(lp);
	dfsbb.perform(LI);
    BasicBlock* hdr = lp->getHeader();
    // this analysis is only applicable to the loops with uniqe latch
    BasicBlock* latch = lp->getLoopLatch();
    if(!latch) return false;
   // BasicBlock* idm = DT->getNode(hdr)->getIDom()->getBlock();
    SmallVector<ValType,32> BBtyStack;
    SmallVector<BasicBlock*,32> MergePoinsStack;
    BBtyStack.clear();
    BBplus* hdrinfo = new BBplus(hdr, ValType::__Const,hdr);
    BBtyStack.push_back(ValType::__Const);
    MergePoinsStack.push_back(hdr);
    BBmap[hdr] = hdrinfo;
    Predicate preds;
    auto* idom = hdr;
    for(auto bb = dfsbb.beginRPO(); bb != dfsbb.endRPO(); ++bb){
		Loop* inner = nullptr;

		// scaping inner loop(s) as they have been processed before the outer
    	if(LI->getLoopFor(*bb) != this->L){
			inner = LI->getLoopFor(*bb);
			while(LI->getLoopFor(*bb) == inner){
				DEBUG(dbgs()<<"passing inner block "<<(*bb)->getName()<<"\t");
				bb++;
			}
			DEBUG(dbgs()<<"musb be exit bb "<<(*bb)->getName()<<"\t idom is "<<MergePoinsStack.back()->getName()<<"\n");

		};

		if( *bb != hdr){
			ValType bType;

			/*
			 * for now we just process loops in standard form (at most 2 predecessors ), should be extended for non-standard form as well
			 */
			int sizeOfPred = 0;
			for(auto prd : predecessors(*bb)) sizeOfPred++;
			if(sizeOfPred > 2) return false;

			if(! isa<BranchInst>((*bb)->getTerminator())) return false;
			BasicBlock* uniqPred = (*bb)->getUniquePredecessor();
			/*if(inner){
				auto innerInf = GetLoopPlusFor(inner);
			}*/
			if(  uniqPred == nullptr){

				if(sizeOfPred > 2) {
					DEBUG(dbgs()<<" reaching block with more than 2 preds : "<<(*bb)->getName()<<"\n");
					for(auto prd : MergePoinsStack)
						DEBUG(dbgs()<<"\t merge point : "<<prd->getName()<<"\n");
					return false;
				}

				bType =BBtyStack.pop_back_val();
				idom = MergePoinsStack.pop_back_val();
				auto idom = findIDom(*bb);
				assert(isa<BranchInst>(idom->getTerminator()) && "can not process call ");
				BranchInst* br = dyn_cast<BranchInst>(idom->getTerminator());
				assert(br->isConditional() && "Idom must have conditional terminator");
				auto cnd = br->getCondition();
				preds.pred = cnd;
			}
			else {
				if(inner){
					uniqPred = inner->getLoopPreheader()? inner->getLoopPreheader() : inner->getLoopPredecessor();
				}
				bType = getType(uniqPred);
				if(! isa<BranchInst>(uniqPred->getTerminator()) ) return false;;
				BranchInst* br = dyn_cast<BranchInst>(uniqPred->getTerminator());
				Value* cnd = nullptr;
				bool jmpCnd = true;
				if(br->isConditional()){
					if(br->getSuccessor(0) != *bb) jmpCnd = false;
					cnd = br->getCondition();
				}
				preds.pred = cnd;
				preds.JumpCnd  = jmpCnd;
				if(getType(br) > bType) bType = getType(br);
			}

			if( !(*bb)->getUniqueSuccessor()) {
				BBtyStack.push_back(bType);
				MergePoinsStack.push_back(*bb);
			}
			auto bbinfo = new BBplus(*bb, bType, idom);
			//auto Idom = DT->getIDom(*bb);
			bbinfo->PRD = preds ;
			BBmap[*bb] = bbinfo;
		}
		DEBUG(dbgs()<<"visiting block "<<(*bb)->getName()<<"\t type :"<<static_cast<int>(getType(*bb))
				<< "\t ImmDom : \t"<<getBlockInfo(*bb)->getIdom()->getName()<<"\t with ");
			auto prd = getBlockInfo(*bb)->PRD;
			if( prd.pred)	DEBUG(dbgs()<<"pred : \t"<<* (prd.pred)<<"\t"<<prd.JumpCnd<<"\n");
			else DEBUG(dbgs()<<"pred : Unconditional \n");
	}

	return true;
}

LoopPlus* LoopPlus::GetLoopPlusFor( Loop* lp) {
	if(!lp) lp = this->L;
	auto lpit = ptrToMap->find(lp);
	if(lpit != ptrToMap->end())
		return lpit->second;
	else return nullptr;
}

ValType LoopPlus::getType( Instruction* inst) const{
	if(!inst) return ValType::__Cx;
	auto mapit = InstMap.find(inst);
	if(mapit != InstMap.end())
		return mapit->second->getType();
	else return ValType::__Cx; // if can not find any info about the instruction then consider its type complex
}

ValType LoopPlus::getType(const BasicBlock* bb) const{
	auto mapit = BBmap.find(bb);
	if(mapit != BBmap.end())
		return mapit->second->getType();
	else return ValType::__Cx;
}

BBplus* LoopPlus::getBlockInfo(const BasicBlock* bb) const{
	auto mapit = BBmap.find(bb);
	if(mapit != BBmap.end())
		return mapit->second;
	else return nullptr;
}

InstPlus* LoopPlus::getInstInfo(const Instruction* inst) const{
	auto mapit = InstMap.find(inst);
	if(mapit != InstMap.end())
		return mapit->second;
	else return nullptr;
}
void LoopPlus::SetType( Instruction* inst,  ValType ty){
	auto mapit = InstMap.find(inst);
	if(mapit == InstMap.end()){
		InstPlus* iplus = new InstPlus(inst , ty);
		InstMap[inst] = iplus;
	}
	else
		mapit->second->SetType(ty);
}

bool LoopPlus::isInvariantofLoop( Value* val, Loop* lp) const{
	if(! lp) lp = this->L;
	if(BranchInst* br = dyn_cast<BranchInst>(val)) val = br->getCondition();
	Instruction* inst = dyn_cast<Instruction>(val);
    if(!inst) return true;
    if(getType(inst) <= ValType::__LI) return true;

    if(isaLoopPhi(inst)) return false;
    for(auto opr = inst->op_begin(); opr != inst->op_end(); ++opr ){
    	if(Instruction* def = dyn_cast<Instruction>(*opr)){
    		if(!lp->contains(def)) continue;
    		if((getType(def) <=  ValType::__LI)) continue;//
    		if(isInvariantofLoop(def, lp)) continue;
    	}

    	if(!lp->isLoopInvariant(*opr)) return false;
    }
    return true;
}

bool LoopPlus::isInductionPhi(PHINode* loopPhi){
	if(! loopPhi) return false;
	if(!isaLoopPhi(loopPhi)) return false;
	Loop* lp = getParentLoop(loopPhi);
	if( lp == nullptr ) return false;
	BasicBlock* Latch = lp->getLoopLatch();
	if(!Latch) return false;

	if(getType(loopPhi) == ValType::__Ind) return true;
	Instruction* lastDef = dyn_cast<Instruction>(loopPhi->getIncomingValueForBlock(Latch));
	std::vector<Instruction*> diverges;
	diverges.push_back(loopPhi);
	llvm::DenseSet<Instruction*> instSet;;
	instSet.insert(loopPhi);
	if(backwardUseDef(lastDef, diverges, lp, instSet, lp->getHeader()) == loopPhi){
		DEBUG(dbgs()<<"induction detected : "<<*loopPhi<<"\n");
		//auto lastinf = getInstInfo(lastDef);
		//printLIVs(lastinf->LiValue);
		//constructInductionStep(loopPhi); /*  moved to inductionoptimization()  method */
		list<Instruction*> worklist;
 		for(auto ScalarInst : instSet)
			worklist.push_back(ScalarInst);
// search for other instructions which can be proved scalar
 		while(!worklist.empty()){
			auto next = worklist.back();
			worklist.pop_back();
			SetType(next, ValType::__Ind);
			DEBUG(dbgs()<<" seting type __Ind : "<<*next<<"\n");
			for(auto use: (next)->users()){
				if (auto useAsInst = dyn_cast<Instruction>(use)){
					bool isScalar  = false;
					if(getType(useAsInst) == ValType::__Ind || (instSet.find(useAsInst) != instSet.end()) ) continue;
					if( getType(useAsInst->getParent()) > ValType::__LI) continue;
					auto op = useAsInst->getOpcode();
					if(useAsInst->isBinaryOp() || op == Instruction::Add || op == Instruction::Sub || op == Instruction::SExt || op == Instruction::Mul
							|| op == Instruction::SDiv || op == Instruction::SRem || op == Instruction::PHI || isa<CmpInst>(useAsInst) ||
							isa<GEPOperator>(useAsInst) || op == Instruction::GetElementPtr){
						isScalar = true;
						for(auto oprd: useAsInst->operand_values()){
							auto  oprinst  = dyn_cast<Instruction>(oprd);
							if(oprinst && lp->contains(oprinst) && (getType(oprinst) > ValType::__Ind) ) isScalar = false;
						}
					}else continue;

					if(isScalar) {
						if(instSet.find(useAsInst) == instSet.end()){
							instSet.insert(useAsInst);
							worklist.push_front(useAsInst);
							//DEBUG(dbgs()<<"\tadding inst to worklist : "<<*useAsInst<<"\n");
						}
					}
				}
			}
		}
		return true;
	}
	return false;
}

void LoopPlus::printLIVs(const LIVal* LiV)const {
	if(!LiV) return;
	if(LiV->val) DEBUG(dbgs()<<LiV->isNegetive<<"\tLIValue :"<<*LiV->val<<"\n");
	if(LiV->LeftDef) printLIVs(LiV->LeftDef);
	if(LiV->RightDef) printLIVs(LiV->RightDef);
}

/*
 *  traversling use-def chain untill reaching a loop-phi which this search started with (success)
 *  OR reaching a condition which disqualifies the Loop Phi for being induction, in this case returns with nullptr
 *
 *  TODO: performance-wise it is not optimized, it does some unnecessary works
 *  TODO: can cover wider range of inductions, for instance if second operand is scalar rather than __LI
 *  TODO: support for mul, shift and other operators, as well as pointers (GEP)
 */
PHINode* LoopPlus::backwardUseDef(Instruction* current, std::vector<Instruction*> &divergPoints, const Loop* ll, DenseSet<Instruction*> &instSet
	, BasicBlock* lastVisitedBB){
	auto lp = this->L;
	//static
	if(getParentLoop(current) != lp){
		//DEBUG(dbgs()<<"can not be ind since passes throw inner Loop "<<*current<<"\n");
		return nullptr;
	}
	auto visitingBB = current->getParent();
	if(getType(visitingBB) > ValType::__LI) return nullptr;
	if(visitingBB != lastVisitedBB){

		divergPoints.push_back(current);
	}
	//DEBUG(dbgs()<<"processing inst :"<<*current<<"\n");
	auto endRec = dyn_cast<PHINode>(current);
	if(isaLoopPhi(endRec) ) return endRec;
	Instruction* next = nullptr;
	Value* LIoperand = nullptr;

	instSet.insert(current);
	LIVal* LiV = new LIVal();
	Livals.push_back(LiV);
	LiV->DefBlock = visitingBB;
	DEBUG(dbgs()<<" current : "<<*current<<"\t");
	if (!(current->getOpcode()== Instruction::Sub || current->getOpcode()== Instruction::Add ||
			isa<CastInst>(current) || isa<SExtInst>(current) || isa<PHINode>(current) || isa<GEPOperator>(current) || isa<SelectInst>(current))) return nullptr;
	/*
	 * handling conditional phis
	 */
	auto phi = dyn_cast<PHINode>(current);
	if( phi && (phi->getNumIncomingValues() > 1) ){
			 //= dyn_cast<PHINode>(current);
		if(isaLoopPhi(phi) ) return phi;

		if(getType(phi) <= ValType::__LI){
			LIoperand = phi;
			DEBUG(dbgs()<<"LI operand "<<*LIoperand<<"\n");
		}
		divergPoints.push_back(phi);
		auto infPhi = getInstInfo(phi);
		assert(infPhi && "no phi info !!!");
		LiV->isPhi = true;
		bool left = true;
		for(auto phiOpr : phi->operand_values()){
			//DEBUG(dbgs()<<"\tphi incom "<<*phiOpr<<"\t");
			auto incom = dyn_cast<Instruction>(phiOpr);
			if(!incom || getType(incom) <= ValType::__LI) continue;
			if(incom &&( backwardUseDef(incom, divergPoints , lp, instSet ,visitingBB) != divergPoints[0]) )return nullptr;
			auto infIncom = getInstInfo(incom);
			assert(infIncom && " no incom info !");
			if(left){
				LiV->LeftDef = infIncom->LiValue ;
				left = false;
			}
			else
				LiV->RightDef = infIncom->LiValue ;
		}
		infPhi->LiValue = LiV;
		divergPoints.pop_back();
		return dyn_cast<PHINode>(divergPoints[0]);
	}

    /*
     * handling iselect
     */
	if(auto cmov = dyn_cast<SelectInst>(current)){
		divergPoints.push_back(cmov);

		if(auto cnd = dyn_cast<Instruction>(cmov->getCondition())){
			if((getType(cnd->getParent()) > ValType::__LI) || !isInvariantofLoop(cnd)) return nullptr;
		}

		if(auto trueVal = dyn_cast<Instruction>(cmov->getTrueValue())){
			if( backwardUseDef(trueVal, divergPoints , lp, instSet ,visitingBB) != divergPoints[0]) return nullptr;
			auto inftrue = getInstInfo(trueVal);
			LiV->LeftDef = inftrue->LiValue;
		}
		if(auto falseVal = dyn_cast<Instruction>(cmov->getFalseValue())){
			if( backwardUseDef(falseVal, divergPoints , lp, instSet ,visitingBB) != divergPoints[0]) return nullptr;
			auto infFalse = getInstInfo(falseVal);
			LiV->RightDef = infFalse->LiValue;
		}
		auto infCmov = getInstInfo(cmov);
		LiV->val = cmov->getCondition();
		infCmov->LiValue = LiV;
		divergPoints.pop_back();
		return dyn_cast<PHINode>(divergPoints[0]);
	}

	/*
	 * handling add/sub , cast, sext, ...
	 */
	for(auto operand : current->operand_values()){
		auto oprInst = dyn_cast<Instruction>(operand);
		//DEBUG(dbgs()<< *operand<<"\n");
		if(  oprInst  &&  (getType(oprInst) > ValType::__LI) && L->contains(oprInst)){

			if (oprInst->getOpcode()== Instruction::Sub || oprInst->getOpcode()== Instruction::Add ||
					isa<CastInst>(oprInst) || isa<SExtInst>(oprInst) || isa<PHINode>(oprInst) ){
				if(next) return nullptr;
				else next = oprInst;
				DEBUG(dbgs()<<"next"<<*next<<"\n");
			}

		}else if(!oprInst || getType(oprInst) <= ValType::__LI ||! this->L->contains(oprInst)){
			if(current->getOpcode()== Instruction::Sub || current->getOpcode()== Instruction::Add){

				DEBUG(dbgs()<<"LI value of "<<*current<<"    \tis   "<<*operand<<"\n");
				LiV->val = operand;
				if(current->getOpcode()== Instruction::Sub) LiV->isNegetive = true;
				else LiV->isNegetive = false;
			}
		}else return nullptr;
	}
	if(!next) return nullptr;
	auto reached = backwardUseDef(next, divergPoints , lp, instSet ,visitingBB);
	if(reached &&  next->getParent() != visitingBB) divergPoints.pop_back();
	auto InfoInst = getInstInfo(current);
	auto infoNext = getInstInfo(next);
	assert(InfoInst && infoNext && " no info !");
	LiV->LeftDef = infoNext->LiValue;
	InfoInst->LiValue = LiV;
	return reached;
}


void LoopPlus::ReleaseMemory(){
	DEBUG(dbgs()<<"\n Loopplus mem release  on\t"<<L->getHeader()->getName() <<" \n");
	for(auto itt = BBmap.begin(); itt != BBmap.end(); ++itt){
		BasicBlock* bb = itt->first;
		if(bb){
		DEBUG(dbgs()<<"delete "<<bb->getName()<<"\n");
		delete itt->second;
		}
	}
	BBmap.clear();
	DEBUG(dbgs()<<"deleting instplus .. \n");
	for(auto itt = InstMap.begin(); itt != InstMap.end(); ++itt){
		//LIVal*
		DEBUG(dbgs()<<*itt->first<<"\t");
		delete itt->second;
	}
	InstMap.clear();

	for(auto LiV : Livals) delete LiV;
	Livals.clear();
}

void IndOptimizer::ReleaseMemory(){

	//DEBUG(dbgs()<<"release mem in function "<<)
	for(auto itt = LoopInfoMap.begin(); itt != LoopInfoMap.end(); ++itt){
		//LIVal*

		delete itt->second;
	}
	LoopInfoMap.clear();

	while(!DeadInsts.empty()){
		auto DI = DeadInsts.back();
		DI->eraseFromParent();
	}

	DeadInsts.clear();

}


/*
 *
 */
