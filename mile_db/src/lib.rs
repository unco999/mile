use std::{
    any::type_name,
    collections::HashMap,
    hash::Hash,
    marker::PhantomData,
    path::Path,
    sync::Arc,
};

use blake3::{Hash as BlakeHash, Hasher as BlakeHasher};
use parking_lot::RwLock;
use redb::{
    backends::InMemoryBackend, Database, DatabaseError, Error as RedbError,
    ReadableTable, TableDefinition,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use thiserror::Error;

/// Errors originating from the storage layer or (de)serialization.
#[derive(Debug, Error)]
pub enum DbError {
    #[error("database error: {0}")]
    Redb(#[from] RedbError),
    #[error("database init error: {0}")]
    Init(#[from] DatabaseError),
    #[error("serialization error: {0}")]
    Serialization(#[from] bincode::Error),
    #[error("hash collision detected for the requested key")]
    HashCollision,
}

/// Fingerprint derived from type information and optional schema descriptors.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct SchemaFingerprint(u128);

impl SchemaFingerprint {
    /// Creates a fingerprint from arbitrary bytes (for example, a serialized schema descriptor).
    pub fn from_bytes(bytes: impl AsRef<[u8]>) -> Self {
        let mut hasher = BlakeHasher::new();
        hasher.update(bytes.as_ref());
        Self::from_hash(hasher.finalize())
    }

    /// Creates a fingerprint that incorporates the type names of the key and value along with the descriptor bytes.
    pub fn for_types<K: 'static, V: 'static>(descriptor: impl AsRef<[u8]>) -> Self {
        let mut hasher = BlakeHasher::new();
        hasher.update(descriptor.as_ref());
        hasher.update(type_name::<K>().as_bytes());
        hasher.update(type_name::<V>().as_bytes());
        Self::from_hash(hasher.finalize())
    }

    fn from_hash(hash: BlakeHash) -> Self {
        let mut buf = [0u8; 16];
        buf.copy_from_slice(&hash.as_bytes()[..16]);
        Self(u128::from_le_bytes(buf))
    }

    /// Returns the fingerprint as a raw `u128` value.
    pub fn as_u128(self) -> u128 {
        self.0
    }

    /// Hex representation, useful when emitting human readable table names.
    pub fn to_hex(self) -> String {
        format!("{:032x}", self.0)
    }
}

/// Builder that can be used to feed additional schema metadata into a [`SchemaFingerprint`].
pub struct SchemaFingerprintBuilder {
    hasher: BlakeHasher,
}

impl SchemaFingerprintBuilder {
    /// Starts a new builder with no data.
    pub fn new() -> Self {
        Self {
            hasher: BlakeHasher::new(),
        }
    }

    /// Seeds the builder with the type names of the provided key/value pair.
    pub fn with_types<K: 'static, V: 'static>(mut self) -> Self {
        self.hasher.update(type_name::<K>().as_bytes());
        self.hasher.update(type_name::<V>().as_bytes());
        self
    }

    /// Adds arbitrary bytes (for example, a hash of the struct field layout).
    pub fn update(&mut self, blob: impl AsRef<[u8]>) {
        self.hasher.update(blob.as_ref());
    }

    /// Completes the builder and produces the fingerprint.
    pub fn finish(self) -> SchemaFingerprint {
        SchemaFingerprint::from_hash(self.hasher.finalize())
    }
}

/// Trait describing how a typed table should be bound into the database.
pub trait TableBinding {
    type Key: Serialize + DeserializeOwned + Eq + Hash + Send + Sync + 'static;
    type Value: Serialize + DeserializeOwned + Send + Sync + 'static;

    /// Provides a descriptor that can encode schema/version metadata.
    /// Override this method when you want the fingerprint to track structural changes.
    fn descriptor() -> &'static str {
        type_name::<Self>()
    }

    /// Computes the fingerprint used to derive a table name in the backing database.
    fn fingerprint() -> SchemaFingerprint {
        SchemaFingerprint::for_types::<Self::Key, Self::Value>(Self::descriptor())
    }
}

/// Lightweight wrapper around a Redb [`Database`] that keeps track of lazily-created tables.
#[derive(Clone)]
pub struct MileDb {
    inner: Arc<Database>,
    table_names: Arc<RwLock<HashMap<SchemaFingerprint, &'static str>>>,
}

impl MileDb {
    /// Creates an in-memory database. Useful for local UI state caches.
    pub fn in_memory() -> Result<Self, DbError> {
        let backend = InMemoryBackend::new();
        let db = Database::builder().create_with_backend(backend)?;
        Ok(Self::from_database(db))
    }

    /// Opens an existing database file, creating it if missing.
    pub fn open_or_create(path: impl AsRef<Path>) -> Result<Self, DbError> {
        let path = path.as_ref();
        let db = if path.exists() {
            Database::open(path)?
        } else {
            Database::create(path)?
        };
        Ok(Self::from_database(db))
    }

    /// Binds a typed table and returns a handle for common CRUD flows.
    pub fn bind_table<B: TableBinding>(&self) -> Result<TableHandle<B>, DbError> {
        let fingerprint = B::fingerprint();
        let name = self.table_name(fingerprint);
        Ok(TableHandle {
            db: Arc::clone(&self.inner),
            table: TableDefinition::new(name),
            fingerprint,
            _binding: PhantomData,
        })
    }

    fn from_database(database: Database) -> Self {
        Self {
            inner: Arc::new(database),
            table_names: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    fn table_name(&self, fingerprint: SchemaFingerprint) -> &'static str {
        if let Some(name) = self.table_names.read().get(&fingerprint).copied() {
            return name;
        }

        let mut guard = self.table_names.write();
        *guard.entry(fingerprint).or_insert_with(|| {
            let name = format!("mile_db::{}", fingerprint.to_hex());
            Box::leak(name.into_boxed_str())
        })
    }
}

/// Handle returned from [`MileDb::bind_table`], parameterised by a [`TableBinding`].
#[derive(Clone)]
pub struct TableHandle<B: TableBinding> {
    db: Arc<Database>,
    table: TableDefinition<'static, u128, Vec<u8>>,
    fingerprint: SchemaFingerprint,
    _binding: PhantomData<B>,
}

impl<B: TableBinding> TableHandle<B> {
    /// Returns the schema fingerprint that backs this table.
    pub fn fingerprint(&self) -> SchemaFingerprint {
        self.fingerprint
    }

    /// Inserts or overwrites a value for the given key.
    pub fn insert(&self, key: &B::Key, value: &B::Value) -> Result<(), DbError> {
        let key_hash = compute_key_hash(key)?;
        let payload = EntryEnvelopeRef { key, value };
        let bytes = bincode::serialize(&payload)?;

        let txn = self.db.begin_write().map_err(map_redb_error)?;
        {
            let mut table = txn.open_table(self.table).map_err(map_redb_error)?;
            table
                .insert(key_hash, &bytes)
                .map_err(map_redb_error)?;
        }
        txn.commit().map_err(map_redb_error)?;
        Ok(())
    }

    /// Fetches a value for the given key.
    pub fn get(&self, key: &B::Key) -> Result<Option<B::Value>, DbError> {
        let key_hash = compute_key_hash(key)?;
        let entry = self.load_entry(key_hash)?;
        match entry {
            Some(found) if found.key == *key => Ok(Some(found.value)),
            Some(_) => Err(DbError::HashCollision),
            None => Ok(None),
        }
    }

    /// Removes the value associated with the key and returns it.
    pub fn remove(&self, key: &B::Key) -> Result<Option<B::Value>, DbError> {
        let key_hash = compute_key_hash(key)?;
        let entry = self.load_entry(key_hash)?;
        match entry {
            Some(found) if found.key == *key => {
                let txn = self.db.begin_write().map_err(map_redb_error)?;
                {
                    let mut table = txn.open_table(self.table).map_err(map_redb_error)?;
                    table.remove(key_hash).map_err(map_redb_error)?;
                }
                txn.commit().map_err(map_redb_error)?;
                Ok(Some(found.value))
            }
            Some(_) => Err(DbError::HashCollision),
            None => Ok(None),
        }
    }

    fn load_entry(&self, key_hash: u128) -> Result<Option<EntryEnvelope<B::Key, B::Value>>, DbError> {
        let txn = self.db.begin_read().map_err(map_redb_error)?;
        let table = txn.open_table(self.table).map_err(map_redb_error)?;
        if let Some(value) = table.get(key_hash).map_err(map_redb_error)? {
            let data = value.value();
            let entry = bincode::deserialize::<EntryEnvelope<B::Key, B::Value>>(&data)?;
            Ok(Some(entry))
        } else {
            Ok(None)
        }
    }
}

#[derive(Serialize)]
struct EntryEnvelopeRef<'a, K, V> {
    key: &'a K,
    value: &'a V,
}

#[derive(Deserialize)]
struct EntryEnvelope<K, V> {
    key: K,
    value: V,
}

fn compute_key_hash<T: Serialize>(value: &T) -> Result<u128, bincode::Error> {
    let bytes = bincode::serialize(value)?;
    Ok(hash_bytes(&bytes))
}

fn hash_bytes(bytes: &[u8]) -> u128 {
    let hash = blake3::hash(bytes);
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&hash.as_bytes()[..16]);
    u128::from_le_bytes(buf)
}

fn map_redb_error<E>(error: E) -> DbError
where
    RedbError: From<E>,
{
    DbError::from(RedbError::from(error))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
    struct UiKey {
        id: u32,
        slot: u8,
    }

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
    struct UiState {
        name: String,
        active: bool,
    }

    struct UiBinding;

    impl TableBinding for UiBinding {
        type Key = UiKey;
        type Value = UiState;

        fn descriptor() -> &'static str {
            "ui_state/v1"
        }
    }

    #[test]
    fn insert_get_remove_roundtrip() -> Result<(), DbError> {
        let db = MileDb::in_memory()?;
        let table = db.bind_table::<UiBinding>()?;
        let key = UiKey { id: 42, slot: 3 };
        let value = UiState {
            name: "InventoryPanel".into(),
            active: true,
        };

        table.insert(&key, &value)?;
        assert_eq!(table.get(&key)?, Some(value.clone()));

        let removed = table.remove(&key)?;
        assert_eq!(removed, Some(value.clone()));
        assert_eq!(table.get(&key)?, None);
        Ok(())
    }

    #[test]
    fn fingerprint_changes_with_descriptor() {
        let a = SchemaFingerprint::for_types::<u32, String>("v1");
        let b = SchemaFingerprint::for_types::<u32, String>("v2");
        assert_ne!(a, b);
    }
}
