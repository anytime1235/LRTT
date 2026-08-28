# Nature Communications 제출본 스냅샷 (2026-08-27)

Overleaf git bridge 클론(paper_natcomm)의 사본. Overleaf 커밋: 10624ec "Update on Overleaf."
paper_edits/ 항목 01–20 전부 반영 완료 시점 (fig6b trainable-parameter 캡션 구절까지).

- main.tex / supplementary.tex / reference.bib + wlscirep.cls, naturemag-doi.bst, jabbrv*
- Figure1–7.png, sup_fig*.png = Overleaf 업로드본 (원본 데이터·스크립트는 LRTT_data/paper_new)

## 예비 문안 (미사용, 리뷰어 요청 시 Methods 에 추가)
fig6b trainable-parameter 상세 산식:
  LR-TT(r) = 48 x (2*768*r) + 39,938 (LayerNorm 25개 + qa_outputs head)
  full FT  = 48 x 768^2 + 39,938 = 28,351,490
  r=1 -> 0.40%, r=64 -> 16.78% (그림 상단축과 일치 검산 완료)
  매핑층 bias 는 양쪽 카운트에서 일관 제외, 코어(C) 타일은 전송으로만 갱신되므로 제외.
