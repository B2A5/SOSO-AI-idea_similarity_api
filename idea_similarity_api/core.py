# 개선된 아이디어 유사도 측정 파이프라인
import pandas as pd
import numpy as np
import faiss
import re
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MinMaxScaler
import joblib

class IdeaSimilarityEngine:
    def __init__(self, csv_path=None):
        # 패키지 내부 데이터 경로 사용
        if csv_path is None:
            import os
            package_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(package_dir, "data", "ideas_sample_1000.csv")
            
            # 파일이 없으면 다른 방법 시도
            if not os.path.exists(csv_path):
                try:
                    import pkg_resources
                    csv_path = pkg_resources.resource_filename('idea_similarity_api', 'data/ideas_sample_1000.csv')
                except:
                    # 마지막 fallback: 상대 경로
                    csv_path = os.path.join(package_dir, "..", "data", "ideas_sample_1000.csv")
        self.df = pd.read_csv(csv_path).fillna("")
        self.embedder = SentenceTransformer("jhgan/ko-sbert-sts")
        self.index = None
        self.emb = None
        self.scaler = MinMaxScaler()
        self._initialize()
    
    def _clean_text(self, txt: str) -> str:
        """향상된 텍스트 전처리"""
        # URL 제거
        txt = re.sub(r"http\S+|www\S+", " ", txt)
        # 특수문자 제거 (한글, 영문, 숫자, 공백만 유지)
        txt = re.sub(r"[^\w가-힣\s]", " ", txt)
        # 중복 공백 제거
        txt = re.sub(r"\s+", " ", txt).strip()
        return txt.lower()
    
    def _calculate_popularity_score(self, likes: int, dislikes: int) -> float:
        """좋아요/싫어요를 기반으로 한 인기도 점수 계산"""
        total = likes + dislikes
        if total == 0:
            return 0.5  # 중립 점수
        return likes / total
    
    def _initialize(self):
        """초기 데이터 로딩 및 임베딩"""
        print("데이터 로딩 중...")
        
        # 텍스트 전처리
        self.df["clean"] = self.df["title"] + " " + self.df["body"]
        self.df["clean"] = self.df["clean"].apply(self._clean_text)
        
        # 인기도 점수 계산
        self.df["popularity_score"] = self.df.apply(
            lambda x: self._calculate_popularity_score(x["좋아요"], x["싫어요"]), axis=1
        )
        
        # 인기도 점수 정규화
        self.df["popularity_normalized"] = self.scaler.fit_transform(
            self.df[["popularity_score"]].values
        ).flatten()
        
        print("임베딩 생성 중...")
        # 임베딩 생성
        self.emb = self.embedder.encode(
            self.df["clean"].tolist(), 
            normalize_embeddings=True
        ).astype("float32")
        
        # FAISS 인덱스 생성
        d = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(self.emb)
        
        print(f"초기화 완료: {len(self.df)}개 아이디어 로드됨")
    
    def find_similar_ideas(self, query: str, top_k: int = 10, 
                          use_popularity: bool = True, 
                          min_similarity: float = 0.3) -> list:
        """
        유사 아이디어 검색
        
        Args:
            query: 검색할 아이디어 텍스트
            top_k: 반환할 상위 결과 수
            use_popularity: 인기도 점수 반영 여부
            min_similarity: 최소 유사도 임계값
        """
        # 쿼리 전처리 및 임베딩
        clean_query = self._clean_text(query)
        query_vec = self.embedder.encode(
            [clean_query], 
            normalize_embeddings=True
        ).astype("float32")
        
        # FAISS 검색
        D, I = self.index.search(query_vec, top_k * 2)  # 더 많은 후보 검색
        
        results = []
        for idx, score in zip(I[0], D[0]):
            if score < min_similarity:
                continue
                
            row = self.df.iloc[idx]
            
            # 인기도 가중치 적용
            final_score = score
            if use_popularity:
                popularity_weight = 0.2  # 인기도 가중치 (20%)
                final_score = score * (1 - popularity_weight) + row["popularity_normalized"] * popularity_weight
            
            results.append({
                "idea_id": row["idea_id"],
                "title": row["title"],
                "body": row["body"][:150] + "..." if len(row["body"]) > 150 else row["body"],
                "similarity_score": round(float(score), 3),
                "final_score": round(float(final_score), 3),
                "likes": row["좋아요"],
                "dislikes": row["싫어요"],
                "popularity_score": round(float(row["popularity_score"]), 3)
            })
            
            if len(results) >= top_k:
                break
        
        # 최종 점수로 정렬
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
    
    def add_new_idea(self, idea_data: dict, top_k: int = 5) -> dict:
        """
        새로운 아이디어 추가 및 유사 아이디어 검색
        
        Args:
            idea_data: {"idea_id": str, "title": str, "body": str, "좋아요": int, "싫어요": int}
        """
        # 기본값 설정
        idea_data.setdefault("좋아요", 0)
        idea_data.setdefault("싫어요", 0)
        idea_data.setdefault("body", "")
        
        # 유사 아이디어 검색
        query = idea_data["title"] + " " + idea_data["body"]
        similar_ideas = self.find_similar_ideas(query, top_k=top_k)
        
        # 새 아이디어를 데이터프레임에 추가 (실제 저장은 선택사항)
        new_clean = self._clean_text(query)
        new_popularity = self._calculate_popularity_score(
            idea_data["좋아요"], idea_data["싫어요"]
        )
        
        new_row = {
            **idea_data,
            "clean": new_clean,
            "popularity_score": new_popularity,
            "popularity_normalized": self.scaler.transform([[new_popularity]])[0][0]
        }
        
        return {
            "new_idea": new_row,
            "similar_ideas": similar_ideas,
            "message": f"'{idea_data['title']}'과 유사한 {len(similar_ideas)}개 아이디어를 찾았습니다."
        }
    
    def get_idea_statistics(self) -> dict:
        """데이터셋 통계 정보"""
        return {
            "total_ideas": len(self.df),
            "avg_likes": self.df["좋아요"].mean(),
            "avg_dislikes": self.df["싫어요"].mean(),
            "most_popular": self.df.loc[self.df["popularity_score"].idxmax()]["title"],
            "least_popular": self.df.loc[self.df["popularity_score"].idxmin()]["title"],
            "popularity_range": {
                "min": self.df["popularity_score"].min(),
                "max": self.df["popularity_score"].max(),
                "mean": self.df["popularity_score"].mean()
            }
        }
    
    def save_model(self, path: str = "./models/idea_similarity_model.pkl"):
        """모델 저장"""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        model_data = {
            "df": self.df,
            "emb": self.emb,
            "index": self.index,
            "scaler": self.scaler
        }
        joblib.dump(model_data, path)
        print(f"모델이 {path}에 저장되었습니다.")
    
    def load_model(self, path: str = "./models/idea_similarity_model.pkl"):
        """모델 로드"""
        model_data = joblib.load(path)
        self.df = model_data["df"]
        self.emb = model_data["emb"]
        self.index = model_data["index"]
        self.scaler = model_data["scaler"]
        print(f"모델이 {path}에서 로드되었습니다.")

# 사용 예시
if __name__ == "__main__":
    # 엔진 초기화
    engine = IdeaSimilarityEngine()
    
    # 통계 정보 출력
    stats = engine.get_idea_statistics()
    print("=== 데이터셋 통계 ===")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # 유사 아이디어 검색 테스트
    test_query = "로봇 바리스타 카페 창업 아이디어"
    print(f"\n=== '{test_query}' 유사 아이디어 검색 ===")
    similar = engine.find_similar_ideas(test_query, top_k=5)
    
    for i, idea in enumerate(similar, 1):
        print(f"{i}. {idea['title']}")
        print(f"   유사도: {idea['similarity_score']}, 최종점수: {idea['final_score']}")
        print(f"   좋아요: {idea['likes']}, 싫어요: {idea['dislikes']}")
        print() 