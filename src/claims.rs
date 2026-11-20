use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition, backends::FileBackend};
use std::{collections::HashSet, io};

const CLAIMS: TableDefinition<u64, ()> = TableDefinition::new("claims");
const MEMORY_ENTRIES: usize = 65_536;

/// Exact process-local deduplication across jobs and reconnects. Long-running
/// miners spill to a temporary file that the OS removes on process exit.
pub(crate) enum Claims {
    Memory(HashSet<u64>),
    File(Database),
}

impl Default for Claims {
    fn default() -> Self {
        Self::Memory(HashSet::new())
    }
}

impl Claims {
    pub fn contains(&self, value: u64) -> io::Result<bool> {
        match self {
            Self::Memory(set) => Ok(set.contains(&value)),
            Self::File(db) => {
                let read = db.begin_read().map_err(io::Error::other)?;
                let table = read.open_table(CLAIMS).map_err(io::Error::other)?;
                Ok(table.get(value).map_err(io::Error::other)?.is_some())
            }
        }
    }

    pub fn insert(&mut self, values: &[u64]) -> io::Result<()> {
        match self {
            Self::Memory(set) if set.len().saturating_add(values.len()) <= MEMORY_ENTRIES => {
                set.extend(values);
                Ok(())
            }
            Self::Memory(set) => {
                let db = Database::builder()
                    .set_cache_size(2 * 1024 * 1024)
                    .create_with_backend(FileBackend::new(tempfile::tempfile()?))
                    .map_err(io::Error::other)?;
                Self::write(&db, set.iter().chain(values))?;
                *self = Self::File(db);
                Ok(())
            }
            Self::File(db) => Self::write(db, values),
        }
    }

    fn write<'a>(db: &Database, values: impl IntoIterator<Item = &'a u64>) -> io::Result<()> {
        let txn = db.begin_write().map_err(io::Error::other)?;
        {
            let mut table = txn.open_table(CLAIMS).map_err(io::Error::other)?;
            for &value in values {
                table.insert(value, ()).map_err(io::Error::other)?;
            }
        }
        txn.commit().map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spilling_preserves_existing_claims_and_accepts_new_ones() {
        let mut claims = Claims::default();
        claims
            .insert(&(0..MEMORY_ENTRIES as u64).collect::<Vec<_>>())
            .unwrap();
        claims.insert(&[u64::MAX, 7]).unwrap();
        assert!(matches!(claims, Claims::File(_)));
        claims.insert(&[u64::MAX - 1]).unwrap();
        for value in [0, 7, MEMORY_ENTRIES as u64 - 1, u64::MAX - 1, u64::MAX] {
            assert!(claims.contains(value).unwrap());
        }
        assert!(!claims.contains(MEMORY_ENTRIES as u64).unwrap());
    }
}
