import { useCallback, useEffect, useState } from "react";

import { ApiError, deleteDocument, type DocumentOut, listDocuments, uploadDocument } from "../api";

export function useDocuments() {
  const [docs, setDocs] = useState<DocumentOut[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setDocs(await listDocuments());
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
  // 這是「跟外部系統同步」，不是同步 setState：setState 發生在 await 之後。
  // React 官方文件把「取遠端資料」列為 effect 的正當用途。
  // oxlint-disable-next-line react/set-state-in-effect
    void refresh();
  }, [refresh]);

  const upload = useCallback(
    async (file: File) => {
      setBusy(true);
      setError(null);
      try {
        await uploadDocument(file);
        await refresh();
        return true;
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e));
        return false;
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  const remove = useCallback(
    async (docId: string) => {
      setBusy(true);
      try {
        await deleteDocument(docId);
        await refresh();
      } catch (e) {
        setError(e instanceof ApiError ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [refresh],
  );

  return { docs, error, busy, upload, remove, refresh, dismissError: () => setError(null) };
}
