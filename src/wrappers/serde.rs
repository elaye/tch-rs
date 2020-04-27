use serde::{
    de::{self, Deserialize, Deserializer, MapAccess, Visitor},
    ser::SerializeMap,
    Serialize, Serializer,
};
use serde_bytes;
use std::{convert::TryInto, fmt};

use crate::{Kind, Tensor};

impl Serialize for Tensor {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let kind = self.kind();
        let size = self.size();

        let n = size.iter().product();

        let data = self.view((n,));

        let data: Vec<u8> = match kind {
            Kind::Uint8 => u8::to_bytes(&data),
            Kind::Int => i32::to_bytes(&data),
            Kind::Int64 => i64::to_bytes(&data),
            Kind::Float => f32::to_bytes(&data),
            Kind::Double => f64::to_bytes(&data),
            k => unimplemented!("Serialization for tensor kind {:?} is not supported", k),
        };

        let data = serde_bytes::ByteBuf::from(data);

        let mut map = serializer.serialize_map(Some(3))?;

        map.serialize_entry("kind", &kind)?;
        map.serialize_entry("size", &size)?;
        map.serialize_entry("data", &data)?;

        map.end()
    }
}

impl<'de> Deserialize<'de> for Tensor {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Kind,
            Size,
            Data,
        };

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Field, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                        formatter.write_str("`kind`, `size` or `data`")
                    }

                    fn visit_str<E>(self, value: &str) -> Result<Field, E>
                    where
                        E: de::Error,
                    {
                        match value {
                            "kind" => Ok(Field::Kind),
                            "size" => Ok(Field::Size),
                            "data" => Ok(Field::Data),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct TensorVisitor;

        impl<'de> Visitor<'de> for TensorVisitor {
            type Value = Tensor;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Tensor")
            }

            fn visit_map<V>(self, mut map: V) -> Result<Tensor, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut kind = None;
                let mut size = None;
                let mut data = None;

                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Kind => {
                            if kind.is_some() {
                                return Err(de::Error::duplicate_field("kind"));
                            }
                            kind = Some(map.next_value()?);
                        }
                        Field::Size => {
                            if size.is_some() {
                                return Err(de::Error::duplicate_field("size"));
                            }
                            size = Some(map.next_value()?);
                        }
                        Field::Data => {
                            if data.is_some() {
                                return Err(de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        }
                    }
                }

                let kind: Kind = kind.ok_or_else(|| de::Error::missing_field("kind"))?;
                let size: Vec<i64> = size.ok_or_else(|| de::Error::missing_field("size"))?;
                let data: serde_bytes::ByteBuf =
                    data.ok_or_else(|| de::Error::missing_field("data"))?;

                let data = data.into_vec();

                let tensor = match kind {
                    Kind::Uint8 => u8::from_bytes(&data),
                    Kind::Int => i32::from_bytes(&data),
                    Kind::Int64 => i64::from_bytes(&data),
                    Kind::Float => f32::from_bytes(&data),
                    Kind::Double => f64::from_bytes(&data),
                    k => unimplemented!("Deserialization for tensor kind {:?} is not supported", k),
                };

                let tensor = tensor.view_(&size);

                Ok(tensor)
            }
        }

        const FIELDS: &'static [&'static str] = &["kind", "size", "data"];
        deserializer.deserialize_struct("Tensor", FIELDS, TensorVisitor)
    }
}

trait ToBytes {
    fn to_bytes(tensor: &Tensor) -> Vec<u8>;
}

macro_rules! to_bytes_num_impl {
    ($t:ident) => {
        impl ToBytes for $t {
            fn to_bytes(tensor: &Tensor) -> Vec<u8> {
                let xs: Vec<$t> = tensor.into();
                let to_bytes_vec = |x: $t| {
                    x.to_be_bytes()
                        .iter()
                        .cloned()
                        .collect::<Vec<u8>>()
                        .into_iter()
                };
                xs.into_iter().flat_map(to_bytes_vec).collect()
            }
        }
    };
}

to_bytes_num_impl!(u8);
to_bytes_num_impl!(i32);
to_bytes_num_impl!(i64);
to_bytes_num_impl!(f32);
to_bytes_num_impl!(f64);

trait FromBytes {
    fn from_bytes(data: &[u8]) -> Tensor;
}

macro_rules! from_bytes_num_impl {
    ($t:ident, $n:literal) => {
        impl FromBytes for $t {
            fn from_bytes(data: &[u8]) -> Tensor {
                let xs: Vec<$t> = data
                    .chunks($n)
                    .map(|bytes| {
                        let bytes: [u8; $n] = bytes.try_into().unwrap();
                        $t::from_be_bytes(bytes)
                    })
                    .collect();
                Tensor::of_slice(&xs)
            }
        }
    };
}

from_bytes_num_impl!(u8, 1);
from_bytes_num_impl!(i32, 4);
from_bytes_num_impl!(i64, 8);
from_bytes_num_impl!(f32, 4);
from_bytes_num_impl!(f64, 8);

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use serde_test::{assert_de_tokens, assert_ser_tokens, Token};

    #[test]
    fn floating_point_tensor() {
        let tensor = Tensor::of_slice(&[1.432, 0., 432.43, 3e12, 3.1e-3, 7.9987]).view((1, 3, 2));

        let tokens = [
            Token::Map { len: Some(3) },
            Token::Str("kind"),
            Token::UnitVariant {
                name: "Kind",
                variant: "Double",
            },
            Token::Str("size"),
            Token::Seq { len: Some(3) },
            Token::I64(1),
            Token::I64(3),
            Token::I64(2),
            Token::SeqEnd,
            Token::Str("data"),
            Token::Bytes(&[
                63, 246, 233, 120, 212, 253, 243, 182, 0, 0, 0, 0, 0, 0, 0, 0, 64, 123, 6, 225, 71,
                174, 20, 123, 66, 133, 211, 239, 121, 128, 0, 0, 63, 105, 101, 43, 211, 195, 97,
                19, 64, 31, 254, 171, 54, 122, 15, 145,
            ]),
            Token::MapEnd,
        ];

        assert_ser_tokens(&tensor, &tokens);
        assert_de_tokens(&tensor, &tokens);
    }
}
