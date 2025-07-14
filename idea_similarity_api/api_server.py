from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager
from .core import IdeaSimilarityEngine

# 전역 엔진 인스턴스
engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """서버 시작/종료 시 이벤트 처리"""
    global engine
    # 서버 시작 시
    try:
        engine = IdeaSimilarityEngine()
        print("아이디어 유사도 엔진이 성공적으로 초기화되었습니다.")
    except Exception as e:
        print(f"엔진 초기화 실패: {e}")
        raise e
    
    yield
    
    # 서버 종료 시 (필요시 정리 작업)

app = FastAPI(
    title="아이디어 유사도 측정 API",
    description="창업 아이디어의 유사도를 측정하고 추천하는 API",
    version="2.0.0",
    lifespan=lifespan
)

# Pydantic 모델들
class IdeaInput(BaseModel):
    idea_id: str = Field(..., description="아이디어 ID")
    title: str = Field(..., description="아이디어 제목", min_length=1, max_length=200)
    body: str = Field("", description="아이디어 상세 내용", max_length=2000)
    좋아요: int = Field(0, description="좋아요 수", ge=0)
    싫어요: int = Field(0, description="싫어요 수", ge=0)

class SearchQuery(BaseModel):
    query: str = Field(..., description="검색할 아이디어 텍스트", min_length=1)
    top_k: int = Field(10, description="반환할 결과 수", ge=1, le=50)
    use_popularity: bool = Field(True, description="인기도 점수 반영 여부")
    min_similarity: float = Field(0.3, description="최소 유사도 임계값", ge=0.0, le=1.0)

class SimilarIdea(BaseModel):
    idea_id: str
    title: str
    body: str
    similarity_score: float
    final_score: float
    likes: int
    dislikes: int
    popularity_score: float

class SearchResponse(BaseModel):
    query: str
    results: List[SimilarIdea]
    total_found: int
    search_params: dict

class AddIdeaResponse(BaseModel):
    new_idea: dict
    similar_ideas: List[SimilarIdea]
    message: str

class StatisticsResponse(BaseModel):
    total_ideas: int
    avg_likes: float
    avg_dislikes: float
    most_popular: str
    least_popular: str
    popularity_range: dict

# API 엔드포인트들
@app.get("/", tags=["Health"])
async def root():
    """API 상태 확인"""
    return {
        "message": "아이디어 유사도 측정 API v2.0",
        "status": "running",
        "engine_loaded": engine is not None
    }

@app.get("/health", tags=["Health"])
async def health_check():
    """상세 헬스 체크"""
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다.")
    
    return {
        "status": "healthy",
        "total_ideas": len(engine.df),
        "model_loaded": True
    }

@app.post("/search", response_model=SearchResponse, tags=["Search"])
async def search_similar_ideas(search_query: SearchQuery):
    """유사 아이디어 검색"""
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다.")
    
    try:
        results = engine.find_similar_ideas(
            query=search_query.query,
            top_k=search_query.top_k,
            use_popularity=search_query.use_popularity,
            min_similarity=search_query.min_similarity
        )
        
        return SearchResponse(
            query=search_query.query,
            results=results,
            total_found=len(results),
            search_params=search_query.dict()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 중 오류 발생: {str(e)}")

@app.post("/add-idea", response_model=AddIdeaResponse, tags=["Ideas"])
async def add_new_idea(idea: IdeaInput):
    """새로운 아이디어 추가 및 유사 아이디어 검색"""
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다.")
    
    try:
        result = engine.add_new_idea(
            idea_data=idea.dict(),
            top_k=5
        )
        
        return AddIdeaResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"아이디어 추가 중 오류 발생: {str(e)}")

@app.get("/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics():
    """데이터셋 통계 정보"""
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다.")
    
    try:
        stats = engine.get_idea_statistics()
        return StatisticsResponse(**stats)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 조회 중 오류 발생: {str(e)}")

@app.get("/ideas/{idea_id}", tags=["Ideas"])
async def get_idea_by_id(idea_id: str):
    """특정 아이디어 정보 조회"""
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다.")
    
    try:
        idea = engine.df[engine.df["idea_id"] == idea_id]
        if idea.empty:
            raise HTTPException(status_code=404, detail=f"아이디어 ID '{idea_id}'를 찾을 수 없습니다.")
        
        return idea.iloc[0].to_dict()
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"아이디어 조회 중 오류 발생: {str(e)}")

@app.get("/ideas", tags=["Ideas"])
async def get_all_ideas(
    limit: int = Query(20, description="반환할 아이디어 수", ge=1, le=100),
    offset: int = Query(0, description="시작 인덱스", ge=0),
    sort_by: str = Query("popularity_score", description="정렬 기준", pattern="^(popularity_score|좋아요|싫어요|title)$")
):
    """모든 아이디어 목록 조회"""
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다.")
    
    try:
        df_sorted = engine.df.sort_values(by=sort_by, ascending=False)
        paginated_df = df_sorted.iloc[offset:offset + limit]
        
        return {
            "ideas": paginated_df.to_dict("records"),
            "total": len(engine.df),
            "limit": limit,
            "offset": offset,
            "sort_by": sort_by
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"아이디어 목록 조회 중 오류 발생: {str(e)}")

@app.post("/save-model", tags=["Model"])
async def save_model(path: str = "./models/idea_similarity_model.pkl"):
    """현재 모델 상태 저장"""
    if engine is None:
        raise HTTPException(status_code=503, detail="엔진이 초기화되지 않았습니다.")
    
    try:
        engine.save_model(path)
        return {"message": f"모델이 {path}에 저장되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 저장 중 오류 발생: {str(e)}")

@app.post("/load-model", tags=["Model"])
async def load_model(path: str = "./models/idea_similarity_model.pkl"):
    """저장된 모델 로드"""
    global engine
    try:
        engine = IdeaSimilarityEngine()
        engine.load_model(path)
        return {"message": f"모델이 {path}에서 로드되었습니다."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"모델 로드 중 오류 발생: {str(e)}")

def main():
    """콘솔 스크립트로 실행하기 위한 메인 함수"""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="아이디어 유사도 측정 API 서버")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트 (기본값: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="서버 포트 (기본값: 8000)")
    parser.add_argument("--reload", action="store_true", help="개발 모드 (자동 리로드)")
    
    args = parser.parse_args()
    
    # 환경 변수에서 설정 가져오기
    host = os.getenv("IDEA_API_HOST", args.host)
    port = int(os.getenv("IDEA_API_PORT", args.port))
    
    print(f"🚀 아이디어 유사도 측정 API 서버를 시작합니다...")
    print(f"📍 서버 주소: http://{host}:{port}")
    print(f"📚 API 문서: http://{host}:{port}/docs")
    print(f"🔧 개발 모드: {args.reload}")
    
    uvicorn.run(
        "idea_similarity_api.api_server:app",
        host=host,
        port=port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 