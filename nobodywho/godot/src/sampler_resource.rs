use godot::global::PropertyHint;
//use godot::builtin::VariantType;
use godot::meta::PropertyHintInfo;
use godot::prelude::*;
use nobodywho::sampler_config;

#[derive(GodotConvert, Var, Export, Debug, Clone, Copy)]
#[godot(via = GString)]
enum SamplerMethodName {
    Greedy,
    DRY,
    TopK,
    TopP,
    MinP,
    XTC,
    TypicalP,
    Temperature,
    MirostatV1,
    MirostatV2,
}

#[derive(GodotClass)]
#[class(tool, base = Resource)]
pub struct NobodyWhoSampler {
    base: Base<Resource>,

    #[export]
    method: SamplerMethodName,

    pub sampler_config: sampler_config::SamplerConfig,
}

macro_rules! property_list {
    ($self:expr,
     base: {$($base_field:ident : $base_type:ty : $property_hint:ident),*},
     methods: {$($variant:ident { $($field:ident : $type:ty),*}),*}
    ) => {
        {
            let base_properties = vec![
                $(
                    godot::meta::PropertyInfo::new_export::<$base_type>(stringify!($base_field)).with_hint_info(PropertyHintInfo { hint: PropertyHint::$property_hint, hint_string: GString::new() }),
                )*
            ];
            let method_properties = match $self.method {
                $(
                    SamplerMethodName::$variant => vec![
                        $(
                            godot::meta::PropertyInfo::new_export::<$type>(stringify!($field)),
                        )*
                    ],
                )*
            };
            let mut result: Vec<godot::meta::PropertyInfo> = base_properties;
            result.extend(method_properties);
            result
        }
    };
}

macro_rules! get_property {
    ($self:expr,
     $property:expr,
     base: {$($base_field:ident : $base_type:ty),*},
     methods: {$($variant:ident { $($variant_field:ident : $variant_type:ty),*}),*}
    ) => {{
        match (&$self.sampler_config.method, $property.to_string().as_str()) {
            (_, "method") => Some(Variant::from($self.method)),
            $(
                (_, stringify!($base_field)) => Some(Variant::from($self.sampler_config.$base_field.clone())),
            )*
            $(
                $(
                    (sampler_config::SamplerMethod::$variant(conf), stringify!($variant_field)) => Some(Variant::from(conf.$variant_field)),
                )*
            )*
            _ => None
        }
    }};
}

macro_rules! set_property {
    ($self:expr,
     $property:expr,
     $value:expr,
     base: {$($base_field:ident : $base_type:ty),*},
     methods: {$($variant:ident { $($variant_field:ident : $variant_type:ty),*}),*}
    ) => {{
        match (&mut $self.sampler_config.method, $property.to_string().as_str()) {
            (_, "method") => {
                let new_method = SamplerMethodName::try_from_variant(&$value).expect("Unexpected: Got invalid sampler method");
                $self.method = new_method;
                $self.sampler_config.method = match new_method {
                    $(
                        SamplerMethodName::$variant => {
                            sampler_config::SamplerMethod::$variant(sampler_config::$variant::default())
                        }
                    )*
                };
                $self.to_gd()
                .upcast::<Object>()
                .notify_property_list_changed(); // <-- CORRECT: Only call this when method changes
            },
            $(
                (_, stringify!($base_field)) => {
                    $self.sampler_config.$base_field = <$base_type>::try_from_variant(&$value)
                    .expect(format!("Unexpected type for {}", stringify!($base_field)).as_str());
                }
            )*
            $(
                $(
                    (sampler_config::SamplerMethod::$variant(conf), stringify!($variant_field)) => {
                        conf.$variant_field = <$variant_type>::try_from_variant(&$value)
                        .expect(format!("Unexpected type for {}", stringify!($variant_field)).as_str());
                    }
                )*
            )*
            (variant, field_name) => godot_warn!("Bad combination of method variant and property name: {:?} {:?}", variant, field_name),
        }
        true
    }};
}

#[godot_api]
impl IResource for NobodyWhoSampler {
    fn init(base: Base<Resource>) -> Self {
        let methodname = match sampler_config::SamplerConfig::default().method {
            sampler_config::SamplerMethod::Greedy(_) => SamplerMethodName::Greedy,
            sampler_config::SamplerMethod::DRY(_) => SamplerMethodName::DRY,
            sampler_config::SamplerMethod::TopK(_) => SamplerMethodName::TopK,
            sampler_config::SamplerMethod::TopP(_) => SamplerMethodName::TopP,
            sampler_config::SamplerMethod::MinP(_) => SamplerMethodName::MinP,
            sampler_config::SamplerMethod::XTC(_) => SamplerMethodName::XTC,
            sampler_config::SamplerMethod::TypicalP(_) => SamplerMethodName::TypicalP,
            sampler_config::SamplerMethod::Temperature(_) => SamplerMethodName::Temperature,
            sampler_config::SamplerMethod::MirostatV1(_) => SamplerMethodName::MirostatV1,
            sampler_config::SamplerMethod::MirostatV2(_) => SamplerMethodName::MirostatV2,
        };
        Self {
            method: methodname,
            sampler_config: sampler_config::SamplerConfig::default(),
            base,
        }
    }

    fn get_property_list(&mut self) -> Vec<godot::meta::PropertyInfo> {
        let mut properties = property_list!(
            self,
            base: {
                penalty_last_n: i32 : NONE,
                penalty_repeat: f32 : NONE,
                penalty_freq: f32 : NONE,
                penalty_present: f32 : NONE,
                use_grammar: bool : NONE,
                gbnf_grammar: GString : MULTILINE_TEXT, // <-- Kept as GString for hint
                use_manual_tool_calling: bool : NONE,
                manual_tool_prefix: GString : MULTILINE_TEXT // <-- Kept as GString for hint
            },
            methods: {
                Greedy { },
                DRY { seed: u32, dry_multiplier: f32, dry_base: f32, dry_allowed_length: i32, dry_penalty_last_n: i32 },
                TopK { seed: u32, top_k: i32 },
                TopP { seed: u32, top_p: f32 },
                MinP { seed: u32, min_keep: u32, min_p: f32 },
                XTC { seed: u32, xtc_probability: f32, xtc_threshold: f32, min_keep: u32 },
                TypicalP { seed: u32, typ_p: f32, min_keep: u32 },
                Temperature { temperature: f32, seed: u32 },
                MirostatV1 { temperature: f32, seed: u32, tau: f32, eta: f32 },
                MirostatV2 { temperature: f32, seed: u32, tau: f32, eta: f32 }
            }
        );

        // --- START: CORRECTED manual_tool_sequence ---
        properties.push(
            godot::meta::PropertyInfo::new_export::<VariantArray>("manual_tool_sequence")
                .with_hint_info(PropertyHintInfo {
                    hint: PropertyHint::ARRAY_TYPE,
                    hint_string: GString::from(format!("{}:", 29)),
                }),
        );
        // --- END: CORRECTED manual_tool_sequence ---

        properties
    }

    fn get_property(&self, property: StringName) -> Option<Variant> {
        let property_str = property.to_string();

        // --- START: manual_tool_sequence GET ---
        if property_str == "manual_tool_sequence" {
            let mut godot_array = VariantArray::new();
            for tool_call in &self.sampler_config.manual_tool_sequence {
                let mut dict = Dictionary::new();
                dict.set("tool_name", tool_call.tool_name.clone());
                dict.set("min_calls", tool_call.min_calls as i64); // Convert to i64 for Godot
                dict.set("max_calls", tool_call.max_calls as i64); // Convert to i64 for Godot

                godot_array.push(&Variant::from(dict));
            }
            return Some(Variant::from(godot_array));
        }
        // --- END: manual_tool_sequence GET ---

        // --- START: GString GET (from previous fix) ---
        if property_str == "gbnf_grammar" {
            return Some(Variant::from(GString::from(
                &self.sampler_config.gbnf_grammar,
            )));
        }
        if property_str == "manual_tool_prefix" {
            return Some(Variant::from(GString::from(
                &self.sampler_config.manual_tool_prefix,
            )));
        }
        // --- END: GString GET ---

        get_property!(
            self, property,
            base: {
                penalty_last_n: i32,
                penalty_repeat: f32,
                penalty_freq: f32,
                penalty_present: f32,
                use_grammar: bool,
                use_manual_tool_calling: bool
            },
            methods: {
                Greedy { },
                DRY { seed: u32, dry_multiplier: f32, dry_base: f32, dry_allowed_length: i32, dry_penalty_last_n: i32 },
                TopK { seed: u32, top_k: i32 },
                TopP { seed: u32, top_p: f32 },
                MinP { seed: u32, min_keep: u32, min_p: f32 },
                XTC { seed: u32, xtc_probability: f32, xtc_threshold: f32, min_keep: u32 },
                TypicalP { seed: u32, typ_p: f32, min_keep: u32 },
                Temperature { temperature: f32, seed: u32 },
                MirostatV1 { temperature: f32, seed: u32, tau: f32, eta: f32 },
                MirostatV2 { temperature: f32, seed: u32, tau: f32, eta: f32 }
            }
        )
    }

    fn set_property(&mut self, property: StringName, value: Variant) -> bool {
        let property_str = property.to_string();

        // --- START: manual_tool_sequence SET ---
        if property_str == "manual_tool_sequence" {
            let godot_array = VariantArray::try_from_variant(&value)
                .expect("Failed to convert Variant to VariantArray for manual_tool_sequence");

            let mut tool_vec = Vec::new();

            for item in godot_array.iter_shared() {
                if item.is_nil() {
                    tool_vec.push(nobodywho::sampler_config::ManualToolCall {
                        tool_name: "new_tool".to_string(),
                        min_calls: 1,
                        max_calls: 1,
                    });
                } else if let Ok(dict) = Dictionary::try_from_variant(&item) {
                    let tool_name = dict
                        .get_or_nil("tool_name")
                        .try_to::<GString>()
                        .map_or(String::new(), |s| s.to_string());

                    let min_calls =
                        dict.get_or_nil("min_calls").try_to::<i64>().unwrap_or(0) as i32;
                    let max_calls =
                        dict.get_or_nil("max_calls").try_to::<i64>().unwrap_or(1) as i32;

                    tool_vec.push(nobodywho::sampler_config::ManualToolCall {
                        tool_name,
                        min_calls,
                        max_calls,
                    });
                }
            }
            self.sampler_config.manual_tool_sequence = tool_vec;

            return true;
        }
        // --- END: manual_tool_sequence SET ---

        // --- START: GString SET (from previous fix) ---
        if property_str == "gbnf_grammar" {
            return match GString::try_from_variant(&value) {
                Ok(gstring) => {
                    self.sampler_config.gbnf_grammar = gstring.to_string();
                    true
                }
                Err(_) => false,
            };
        }
        if property_str == "manual_tool_prefix" {
            return match GString::try_from_variant(&value) {
                Ok(gstring) => {
                    self.sampler_config.manual_tool_prefix = gstring.to_string();
                    true
                }
                Err(_) => false,
            };
        }
        // --- END: GString SET ---

        set_property!(
            self, property, value,
            base: {
                penalty_last_n: i32,
                penalty_repeat: f32,
                penalty_freq: f32,
                penalty_present: f32,
                use_grammar: bool,
                use_manual_tool_calling: bool
            },
            methods: {
                Greedy { },
                DRY { seed: u32, dry_multiplier: f32, dry_base: f32, dry_allowed_length: i32, dry_penalty_last_n: i32 },
                TopK { seed: u32, top_k: i32 },
                TopP { seed: u32, top_p: f32 },
                MinP { seed: u32, min_keep: u32, min_p: f32 },
                XTC { seed: u32, xtc_probability: f32, xtc_threshold: f32, min_keep: u32 },
                TypicalP { seed: u32, typ_p: f32, min_keep: u32 },
                Temperature { temperature: f32, seed: u32 },
                MirostatV1 { temperature: f32, seed: u32, tau: f32, eta: f32 },
                MirostatV2 { temperature: f32, seed: u32, tau: f32, eta: f32 }
            }
        )
    }
}
