use godot::global::PropertyHint;
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
                    godot::meta::PropertyInfo::new_export::<$base_type>(stringify!($base_field))
                        .with_hint_info(PropertyHintInfo { 
                            hint: PropertyHint::$property_hint, 
                            hint_string: GString::new() 
                        }),
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
                    (sampler_config::SamplerMethod::$variant(conf), stringify!($variant_field)) => 
                        Some(Variant::from(conf.$variant_field)),
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
                let new_method = SamplerMethodName::try_from_variant(&$value)
                    .expect("Unexpected: Got invalid sampler method");
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
                    .notify_property_list_changed();
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
            (variant, field_name) => {
                godot_warn!("Bad combination of method variant and property name: {:?} {:?}", variant, field_name);
            },
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
                gbnf_grammar: GString : MULTILINE_TEXT,
                use_manual_tool_calling: bool : NONE,
                manual_tool_prefix: GString : MULTILINE_TEXT
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

        // FIXED: Proper array type hint for Dictionary arrays
        // Format is "type_hint:hint_string" where type_hint is the VariantType enum value
        // 27 = TYPE_DICTIONARY
        properties.push(
            godot::meta::PropertyInfo::new_export::<Array<Dictionary>>("manual_tool_sequence")
                .with_hint_info(PropertyHintInfo {
                    hint: PropertyHint::ARRAY_TYPE,
                    hint_string: GString::from("27:"), // Dictionary type
                }),
        );

        properties
    }

    fn get_property(&self, property: StringName) -> Option<Variant> {
        let property_str = property.to_string();

        if property_str == "manual_tool_sequence" {
            let mut godot_array = Array::<Dictionary>::new();
            for tool_call in &self.sampler_config.manual_tool_sequence {
                let mut dict = Dictionary::new();
                dict.set("tool_name", tool_call.tool_name.clone());
                dict.set("min_calls", tool_call.min_calls as i64);
                dict.set("max_calls", tool_call.max_calls as i64);
                godot_array.push(&dict);
            }
            return Some(godot_array.to_variant());
        }

        get_property!(
            self, property,
            base: {
                penalty_last_n: i32,
                penalty_repeat: f32,
                penalty_freq: f32,
                penalty_present: f32,
                use_grammar: bool,
                gbnf_grammar: String,
                use_manual_tool_calling: bool,
                manual_tool_prefix: String
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

        if property_str == "manual_tool_sequence" {
            // Try to convert to Array<Dictionary> first (type-safe)
            if let Ok(godot_array) = Array::<Dictionary>::try_from_variant(&value) {
                let mut tool_vec = Vec::new();

                for dict in godot_array.iter_shared() {
                    let tool_name = dict
                        .get_or_nil("tool_name")
                        .try_to::<GString>()
                        .map(|gstr| gstr.to_string())
                        .unwrap_or_else(|_| "new_tool".to_string());
                    
                    let min_calls = dict
                        .get_or_nil("min_calls")
                        .try_to::<i64>()
                        .unwrap_or(1) as i32;
                    
                    let max_calls = dict
                        .get_or_nil("max_calls")
                        .try_to::<i64>()
                        .unwrap_or(1) as i32;
                    
                    tool_vec.push(nobodywho::sampler_config::ManualToolCall {
                        tool_name,
                        min_calls,
                        max_calls,
                    });
                }
                
                self.sampler_config.manual_tool_sequence = tool_vec;
                return true;
            }
            
            // Fallback to VariantArray for compatibility
            if let Ok(godot_array) = VariantArray::try_from_variant(&value) {
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
                            .map(|gstr| gstr.to_string())
                            .unwrap_or_else(|_| "new_tool".to_string());
                        
                        let min_calls = dict
                            .get_or_nil("min_calls")
                            .try_to::<i64>()
                            .unwrap_or(1) as i32;
                        
                        let max_calls = dict
                            .get_or_nil("max_calls")
                            .try_to::<i64>()
                            .unwrap_or(1) as i32;
                        
                        tool_vec.push(nobodywho::sampler_config::ManualToolCall {
                            tool_name,
                            min_calls,
                            max_calls,
                        });
                    } else {
                        godot_warn!(
                            "Item in manual_tool_sequence was not a Dictionary: {:?}",
                            item
                        );
                    }
                }
                
                self.sampler_config.manual_tool_sequence = tool_vec;
                return true;
            }
            
            godot_warn!("Failed to parse manual_tool_sequence as Array");
            return false;
        }

        set_property!(
            self, property, value,
            base: {
                penalty_last_n: i32,
                penalty_repeat: f32,
                penalty_freq: f32,
                penalty_present: f32,
                use_grammar: bool,
                gbnf_grammar: String,
                use_manual_tool_calling: bool,
                manual_tool_prefix: String
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